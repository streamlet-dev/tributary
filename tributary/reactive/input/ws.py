from ujson import loads as load_json
from websocket import create_connection
from ..base import _wrap
from ...base import StreamNone, StreamEnd
from ...thread import run


def WebSocket(url, *args, **kwargs):
    return AsyncWebSocket(url, *args, **kwargs)


def AsyncWebSocket(url, json=False, wrap=False):
    async def _listen(url, json, wrap):
        ws = create_connection(url)
        for x in run(ws.recv):
            if isinstance(x, StreamNone):
                continue
            elif not x or isinstance(x, StreamEnd):
                break

            if json:
                x = load_json(x)
            if wrap:
                x = [x]
            yield x

    return _wrap(_listen, dict(url=url, json=json, wrap=wrap), name='WebSocket')
