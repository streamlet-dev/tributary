import aiohttp
import json as JSON
from .output import _OUTPUT_GRAPHVIZSHAPE
from ..node import Node
from ...base import StreamNone, StreamEnd


def WebSocket(node, url='', json=False, wrap=False, field=None, response=False, response_timeout=1):
    '''Connect to websocket and send data

    Args:
        node (Node): input tributary
        url (str): websocket url to connect to
        json (bool): dump data as json
        wrap (bool): wrap result in a list
    '''
    async def _send(data, url=url, json=json, wrap=wrap, field=field, response=response, response_timeout=response_timeout):
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
                    x = '{}'
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    x = '{}'
            else:
                x = '{}'

        await session.close()

        if json:
            x = JSON.loads(x)

        if field:
            x = x[field]

        if wrap:
            x = [x]

        return x

    ret = Node(foo=_send, name='WebSocket', inputs=1, graphvizshape=_OUTPUT_GRAPHVIZSHAPE)
    node >> ret
    return ret
