import functools
import ujson
import websockets
from ..base import _wrap
from ...base import StreamNone, StreamEnd


def AsyncWebSocket(foo, foo_kwargs=None, url='', json=False, wrap=False, field=None, response=False):
    '''Connect to websocket and send data

    Args:
        foo (callable): input stream
        foo_kwargs (dict): kwargs for the input stream
        url (str): websocket url to connect to
        json (bool): dump data as json
        wrap (bool): wrap result in a list
    '''
    foo_kwargs = foo_kwargs or {}
    foo = _wrap(foo, foo_kwargs)

    async def _send(foo, url, json=False, wrap=False, field=None, response=False):
        async with websockets.connect(url) as websocket:
            async for data in foo():
                if isinstance(data, StreamNone):
                    continue
                elif not data or isinstance(data, StreamEnd):
                    break

                if wrap:
                    data = [data]
                if json:
                    data = ujson.dumps(data)

                await websocket.send(data)

                if response:
                    msg = await websocket.recv()

                else:
                    msg = '{}'

                if json:
                    msg = json.loads(msg)

                if field:
                    msg = msg[field]

                if wrap:
                    msg = [msg]

                yield msg

    return _wrap(_send, dict(foo=foo, url=url, json=json, wrap=wrap, field=field, response=response), name='WebSocket')


@functools.wraps(AsyncWebSocket)
def WebSocket(foo, foo_kwargs=None, *args, **kwargs):
    return AsyncWebSocket(foo, foo_kwargs, *args, **kwargs)
