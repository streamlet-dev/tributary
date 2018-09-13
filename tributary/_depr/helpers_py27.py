import trollius as asyncio
from trollius import From, Return
# import websockets  # not supported in python2
from requests_futures.sessions import FuturesSession
from socketIO_client_nexus import SocketIO, BaseNamespace


class GenBase(object):
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

    @asyncio.coroutine
    def run(self):
        while True:
            item = yield From(self.getData())
            if item is None:
                return
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
        self.psp = psp
        self.first = True

    @asyncio.coroutine
    def fetch(self):
        session = FuturesSession()
        dat = session.get(self.url).result().json()
        if self.field:
            dat = dat[self.field]
        if self.records is False:
            dat = [dat]
        raise Return(dat)

    @asyncio.coroutine
    def getData(self):
        if self.first:
            self.first = False
            dat = yield From(self.fetch())
            raise Return(dat)
        else:
            if self.repeat > 0:
                yield From(asyncio.sleep(self.repeat))
                dat = yield From(self.fetch())
                raise Return(dat)
            else:
                raise Return(None)


# class WSHelper(GenBase):
#     def __init__(self, psp, url, send=None, records=True):
#         self.validate(url, 'ws')
#         self.__type = 'ws'
#         self.url = url
#         self.send = send
#         self.records = records
#         self.psp = psp

#     @asyncio.coroutine
#     def getData(self):
#         websocket = yield From(websockets.connect(self.url))
#         if self.send:
#             yield From(websocket.send(self.send))

#         data = yield From(websocket.recv())

#         if self.records is False:
#             raise Return([data])
#         else:
#             raise Return(data)


class SIOHelper(GenBase):
    def __init__(self, psp, url, send=None, channel='', records=False):
        self.validate(url, 'sio')
        self.__type = 'sio'
        self.url = url
        self.send = send
        self.channel = channel
        self.records = records

        self._data = []

    @asyncio.coroutine
    def getData(self):
        # FIXME
        class Namespace(BaseNamespace):
            def on_connect(self, *data):
                pass

            def on_disconnect(self, *data):
                pass

            def on_message(self, data):
                return data

        self.socketIO = SocketIO(self.url, 443)  # needs base url
        namespace = self.socketIO.define(Namespace, self.url)  # needs path in url
        if self.send:
            namespace.emit(*self.send)
        self.socketIO.wait()


def type_to_helper(type):
    if type.startswith('http'):
        return HTTPHelper
    elif type.startswith('ws'):
        # return WSHelper
        pass
    elif type.startswith('sio'):
        return SIOHelper
