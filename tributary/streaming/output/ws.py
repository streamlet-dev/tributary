import json as JSON
import websockets
from ..base import Node
from ...base import StreamNone, StreamEnd


class WebSocket(Node):
    '''Connect to websocket and send data

    Args:
        foo (callable): input stream
        foo_kwargs (dict): kwargs for the input stream
        url (str): websocket url to connect to
        json (bool): dump data as json
        wrap (bool): wrap result in a list
    '''

    def __init__(self, node, url='', json=False, wrap=False, field=None, response=False):
        self._websocket = websockets.connect(url)

        async def _send(data, url=url, json=json, wrap=wrap, field=field, response=response):
            if isinstance(data, (StreamNone, StreamEnd)):
                return data

            if wrap:
                data = [data]
            if json:
                data = JSON.dumps(data)

            await self._websocket.send(data)

            if response:
                msg = await self._websocket.recv()

            else:
                msg = '{}'

            if json:
                msg = JSON.loads(msg)

            if field:
                msg = msg[field]

            if wrap:
                msg = [msg]

            return msg

        super().__init__(foo=_send, name='WebSocket', inputs=1)
        node._downstream.append((self, 0))
        self._upstream.append(node)
