from socketIO_client_nexus import SocketIO as SIO
try:
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse
from ..base import _wrap


def SocketIO(url, *args, **kwargs):
    return AsyncSocketIO(url, *args, **kwargs)


def AsyncSocketIO(url, channel='', field='', sendinit=None, json=False, wrap=False, interval=1):
    o = urlparse(url)
    socketIO = SIO(o.scheme + '://' + o.netloc, o.port)
    if sendinit:
        socketIO.emit(sendinit)

    async def _sio(url, channel, field='', json=False, wrap=False, interval=1):
        while True:
            _data = []
            socketIO.on(channel, lambda data: _data.append(data))
            socketIO.wait(seconds=interval)
            for msg in _data:
                # FIXME clear _data
                if json:
                    msg = json.loads(msg)

                if field:
                    msg = msg[field]

                if wrap:
                    msg = [msg]

                yield msg

    return _wrap(_sio, dict(url=url, channel=channel, field=field, json=json, wrap=wrap, interval=interval), name='SocketIO')
