import aiohttp
import json as JSON
from .output import Foo
from ..node import Node
from ...base import StreamNone, StreamEnd


class WebSocket(Foo):
    """Connect to websocket and send data

    Args:
        node (Node): input tributary
        url (str): websocket url to connect to
        json (bool): dump data as json
        wrap (bool): wrap result in a list
    """

    def __init__(
        self,
        node,
        url,
        json=False,
        wrap=False,
        field=None,
        response=False,
        response_timeout=1,
    ):
        async def _send(
            data,
            url=url,
            json=json,
            wrap=wrap,
            field=field,
            response=response,
            response_timeout=response_timeout,
        ):
            if isinstance(data, (StreamNone, StreamEnd)):
                return data

            if wrap:
                data = [data]
            if json:
                data = JSON.dumps(data)

            session = aiohttp.ClientSession()
            async with session.ws_connect(url) as ws:
                await ws.send_str(data)

                if response:
                    msg = await ws.receive(response_timeout)
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        x = msg.data
                    elif msg.type == aiohttp.WSMsgType.CLOSED:
                        x = "{}"
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        x = "{}"
                else:
                    x = "{}"

            await session.close()

            if json:
                x = JSON.loads(x)

            if field:
                x = x[field]

            if wrap:
                x = [x]

            return x

        super().__init__(foo=_send, name="WebSocket", inputs=1)
        node >> self


class WebSocketServer(Foo):
    """Host a websocket server and send data to clients

    Args:
        node (Node): input tributary
        json (bool): dump data as json
        wrap (bool): wrap result in a list
    """

    def __init__(
        self,
        node,
        json=False,
        wrap=False,
        field=None,
        response=False,
        response_timeout=1,
    ):
        async def _send(
            data,
            json=json,
            wrap=wrap,
            field=field,
        ):
            raise NotImplementedError()

        super().__init__(foo=_send, name="WebSocketServer", inputs=1)
        node >> self


Node.websocket = WebSocket
Node.websocketServer = WebSocketServer
