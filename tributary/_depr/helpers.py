import asyncio
import json
import websockets
import tornado.httpclient
import threading
from urllib.parse import urlparse
from socketIO_client_nexus import SocketIO, BaseNamespace


class GenBase(object):
    def __init__(self, psp):
        self.psp = psp

    def validate(self, url, type):
        if type == 'http':
            if not url.startswith('http://') and not url.startswith('https://'):
                raise Exception('Invalid url for type http: %s' % url)
        elif type == 'ws':
            if not url.startswith('ws://') and not url.startswith('wss://'):
                raise Exception('Invalid url for type ws: %s' % url)
        elif type == 'sio':
            if not url.startswith('sio://'):
                raise Exception('Invalid url for type socketio: %s' % url)

    async def run(self):
        async for item in self.getData():
            self.psp.update(item)

    def start(self):
        loop = asyncio.get_event_loop()

        try:
            if not loop.is_running():
                loop.run_until_complete(self.run())
            else:
                asyncio.ensure_future(self.run(), loop=loop)
        except KeyboardInterrupt:
            loop.close()


class HTTPHelper(GenBase):
    def __init__(self, psp, url, field='', records=True, repeat=-1):
        self.validate(url, 'http')
        self.__type = 'http'
        self.url = url
        self.field = field
        self.records = records
        self.repeat = repeat
        super(HTTPHelper, self).__init__(psp)

    async def getData(self):
        client = tornado.httpclient.AsyncHTTPClient()
        dat = await client.fetch(self.url)
        dat = json.loads(dat.body)

        if self.field:
            dat = dat[self.field]
        if self.records is False:
            dat = [dat]
        yield dat

        while(self.repeat >= 0):
            await asyncio.sleep(self.repeat)
            dat = await client.fetch(self.url)
            dat = json.loads(dat.body)
            if self.field:
                dat = dat[self.field]
            if self.records is False:
                dat = [dat]
            yield dat


class WSHelper(GenBase):
    def __init__(self, psp, url, send=None, records=True):
        self.validate(url, 'ws')
        self.__type = 'ws'
        self.url = url
        self.send = send
        self.records = records
        super(WSHelper, self).__init__(psp)

    async def getData(self):
        async with websockets.connect(self.url) as websocket:
            if self.send:
                await websocket.send(self.send)

            data = await websocket.recv()
            raise Exception(data)
            if self.records is False:
                yield [data]
            else:
                yield data


class SIOHelper(GenBase):
    def __init__(self, psp, url, send=None, channel='', records=False):
        self.validate(url, 'sio')
        self.__type = 'sio'
        self.url = url.replace('sio://', '')
        self.send = send
        self.channel = channel
        self.records = records

        self._data = []

        o = urlparse(self.url)

        self.socketIO = SocketIO(o.scheme + '://' + o.netloc, o.port)
        self.socketIO.on(self.channel, lambda data: self._data.append(data))
        self.url = o.path

        super(SIOHelper, self).__init__(psp)

    async def getData(self):
        if self.send:
            self.socketIO.emit(*self.send)
        t = threading.Thread(target=self.socketIO.wait)
        t.start()

        while 1:
            if self._data:
                c = 0
                for item in self._data:
                    c += 1
                    yield item
                self._data = self._data[:c]

            else:
                await asyncio.sleep(1)


def type_to_helper(type):
    if type.startswith('http'):
        return HTTPHelper
    elif type.startswith('ws'):
        return WSHelper
    elif type.startswith('sio'):
        return SIOHelper
