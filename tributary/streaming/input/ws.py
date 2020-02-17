import json as JSON
import websockets

from .input import Foo
from ...base import StreamNone, StreamEnd


class WebSocket(Foo):
    '''Connect to websocket and yield back results

    Args:
        url (str): websocket url to connect to
        json (bool): load websocket data as json
        wrap (bool): wrap result in a list
    '''

    def __init__(self, url, json=False, wrap=False):
        async def _listen(url=url, json=json, wrap=wrap):
            async with websockets.connect(url) as websocket:
                async for x in websocket:
                    if isinstance(x, StreamNone):
                        continue
                    elif not x or isinstance(x, StreamEnd):
                        break

                    if json:
                        x = JSON.loads(x)
                    if wrap:
                        x = [x]
                    yield x

        super().__init__(foo=_listen)
        self._name = 'WebSocket'
