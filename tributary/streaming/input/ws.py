import aiohttp
import json as JSON

from .input import Foo


class WebSocket(Foo):
    """Connect to websocket and yield back results

    Args:
        url (str): websocket url to connect to
        json (bool): load websocket data as json
        wrap (bool): wrap result in a list
    """

    def __init__(self, url, json=False, wrap=False):
        async def _listen(url=url, json=json, wrap=wrap):
            session = aiohttp.ClientSession()
            async with session.ws_connect(url) as ws:

                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        x = msg.data
                        if json:
                            x = JSON.loads(x)
                        if wrap:
                            x = [x]
                        yield x
                    elif msg.type == aiohttp.WSMsgType.CLOSED:
                        break
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        break

        super().__init__(foo=_listen)
        self._name = "WebSocket"


class WebSocketServer(Foo):
    """Host a websocket server and yield data as sent

    Args:
        json (bool): load websocket data as json
        wrap (bool): wrap result in a list
    """

    def __init__(self, json=False, wrap=False):
        async def _listen(json=json, wrap=wrap):
            raise NotImplementedError()

        super().__init__(foo=_listen)
        self._name = "WebSocketServer"
