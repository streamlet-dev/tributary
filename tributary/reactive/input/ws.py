from websocket import create_connection
from ..base import _wrap


def WebSocket(url, *args, **kwargs):
    return SyncWebSocket(url, *args, **kwargs)


def SyncWebSocket(url, json=False, wrap=False):
    def _listen(url):
        ws = create_connection("ws://localhost:8899")
        while True:
            msg = ws.recv()
            if msg is None:
                break
            if json:
                msg = json.loads(msg)
            if wrap:
                msg = [msg]
            yield msg

    return _wrap(_listen, dict(url=url), name='WebSocket')
