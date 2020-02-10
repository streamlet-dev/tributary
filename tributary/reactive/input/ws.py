import functools
import websockets
from ujson import loads as load_json
from ..base import _wrap
from ...base import StreamNone, StreamEnd



def AsyncWebSocket(url, json=False, wrap=False):
    '''Connect to websocket and yield back results

    Args:
        url (str): websocket url to connect to
        json (bool): load websocket data as json
        wrap (bool): wrap result in a list
    '''
    async def _listen(url, json, wrap):
        async with websockets.connect(url) as websocket:
            async for x in websocket:
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


@functools.wraps(AsyncWebSocket)
def WebSocket(url, *args, **kwargs):
    return AsyncWebSocket(url, *args, **kwargs)
