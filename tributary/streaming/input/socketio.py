import json as JSON
from socketIO_client_nexus import SocketIO as SIO
from urllib.parse import urlparse

from .input import Foo


class SocketIO(Foo):
    '''Connect to socketIO server and yield back results

    Args:
        url (str): url to connect to
        channel (str): socketio channel to connect through
        field (str): field to index result by
        sendinit (list): data to send on socketio connection open
        json (bool): load websocket data as json
        wrap (bool): wrap result in a list
        interval (int): socketio wai interval
    '''

    def __init__(self, url, channel='', field='', sendinit=None, json=False, wrap=False, interval=1):
        o = urlparse(url)
        socketIO = SIO(o.scheme + '://' + o.netloc, o.port)
        if sendinit:
            socketIO.emit(sendinit)

        async def _sio(url=url, channel=channel, field=field, json=json, wrap=wrap, interval=interval):
            while True:
                _data = []
                socketIO.on(channel, lambda data: _data.append(data))
                socketIO.wait(seconds=interval)
                for msg in _data:
                    # FIXME clear _data
                    if json:
                        msg = JSON.loads(msg)

                    if field:
                        msg = msg[field]

                    if wrap:
                        msg = [msg]

                    yield msg
        super().__init__(foo=_sio)
        self._name = 'SocketIO'
