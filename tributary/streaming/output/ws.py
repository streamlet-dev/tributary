import json as JSON
import websockets
from .output import _OUTPUT_GRAPHVIZSHAPE
from ..node import Node
from ...base import StreamNone, StreamEnd


def WebSocket(node, url='', json=False, wrap=False, field=None, response=False):
    '''Connect to websocket and send data

    Args:
        foo (callable): input stream
        foo_kwargs (dict): kwargs for the input stream
        url (str): websocket url to connect to
        json (bool): dump data as json
        wrap (bool): wrap result in a list
    '''
    async def _send(data, url=url, json=json, wrap=wrap, field=field, response=response):
        if isinstance(data, (StreamNone, StreamEnd)):
            return data

        if wrap:
            data = [data]
        if json:
            data = JSON.dumps(data)

        await ret._websocket.send(data)

        if response:
            msg = await ret._websocket.recv()

        else:
            msg = '{}'

        if json:
            msg = JSON.loads(msg)

        if field:
            msg = msg[field]

        if wrap:
            msg = [msg]

        return msg

    ret = Node(foo=_send, name='WebSocket', inputs=1, graphvizshape=_OUTPUT_GRAPHVIZSHAPE)
    ret._websocket = websockets.connect(url)

    node.downstream().append((ret, 0))
    ret.upstream().append(node)
    return ret
