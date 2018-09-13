from socketIO_client_nexus import SocketIO as SIO
try:
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse
from ..base import _wrap


def SocketIO(url, *args, **kwargs):
    return SyncSocketIO(url, *args, **kwargs)


def SyncSocketIO(url, channel='', field='', sendinit=None, json=False, wrap=False):
    o = urlparse(url)
    socketIO = SIO(o.scheme + '://' + o.netloc, o.port)
    if sendinit:
        socketIO.emit(sendinit)

    def _sio(url, channel, field='', json=False, wrap=False):
        while True:
            _data = []
            socketIO.on(channel, lambda data: _data.append(data))
            socketIO.wait(seconds=1)
            for msg in _data:
                if json:
                    msg = json.loads(msg)

                if field:
                    msg = msg[field]

                if wrap:
                    msg = [msg]

                yield msg

    return _wrap(_sio, dict(url=url, channel=channel, field=field, json=json, wrap=wrap), name='SocketIO')
