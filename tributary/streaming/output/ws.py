import asyncio
import aiohttp
import json as JSON
from collections import deque
from aiohttp import web
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
        binary (bool): send_bytes instead of send_str
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
        binary=False,
    ):
        async def _send(
            data,
            url=url,
            json=json,
            wrap=wrap,
            field=field,
            response=response,
            response_timeout=response_timeout,
            binary=binary,
        ):
            if isinstance(data, (StreamNone, StreamEnd)):
                return data

            if wrap:
                data = [data]
            if json:
                data = JSON.dumps(data)

            session = aiohttp.ClientSession()
            async with session.ws_connect(url) as ws:
                if binary:
                    await ws.send_bytes(data)
                else:
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
    """Host a websocket server and stream in the data

    Args:
        path (str): route on which to host ws server
        json (bool): load http content data as json
        wrap (bool): wrap result in a list
        field (str): field to index result by
        snapshot (bool): maintain history and provide a snapshot on first request
        server (Optional[aiohttp.web.Application]): aiohttp application to install route
        run (Optional[bool]): run the web app right after construction
        host (Optional[str]): if running the web app, which ip to listen on (default 127.0.0.1)
        port (Optional[int]): if running the web app, port to listen on
        request_handler (Optional[callable]): custom handler to process the request from client
        response_handler (Optional[callable]): custom handler to manage the response sent to client
        binary (bool): send_bytes instead of send_str
    """

    def __init__(
        self,
        node,
        path="/",
        json=False,
        wrap=False,
        field=None,
        snapshot=True,
        server=None,
        run=True,
        host="127.0.0.1",
        port=8080,
        request_handler=None,
        response_handler=None,
        binary=False,
    ):
        # instantiate server if not existing
        server = server or web.Application()

        # capture history
        self._history = deque()

        # create queue map for clients
        self._queue_map = {}

        # http server handler
        async def _handler(
            request,
            queue_map=self._queue_map,
            history=self._history,
            request_handler=request_handler,
            response_handler=response_handler,
            binary=binary,
        ):
            ws = web.WebSocketResponse()
            await ws.prepare(request)

            if ws not in queue_map:
                # create a queue for this client
                queue_map[ws] = asyncio.Queue()

                # send history if snapshotting
                for data in history:
                    if response_handler and callable(response_handler):
                        data = await response_handler(request, data)
                    if binary:
                        await ws.send_bytes(data)
                    else:
                        await ws.send_str(data)

                queue = queue_map[ws]

                try:
                    while not ws.closed:
                        # put the request into the queue
                        data = await queue.get()

                        # TODO move this?
                        if request_handler and callable(request_handler):
                            data = await request_handler(request)

                        # if custom response handler is given, use that to determine response
                        if response_handler and callable(response_handler):
                            data = await response_handler(request, data)
                            if binary:
                                await ws.send_bytes(data)
                            else:
                                await ws.send_str(data)

                        elif response_handler and isinstance(
                            response_handler, (str, bytes)
                        ):
                            if binary:
                                await ws.send_bytes(response_handler)
                            else:
                                await ws.send_str(response_handler)
                        else:
                            # just put an ok with data
                            await ws.send_str(JSON.dumps(data))
                finally:
                    # remove from queue
                    queue_map.pop(ws)

        # tributary node handler
        async def _req(
            data,
            json=json,
            wrap=wrap,
            field=field,
            snapshot=snapshot,
            queue_map=self._queue_map,
            history=self._history,
        ):
            if json:
                data = JSON.dumps(data)

            if field:
                data = data[field]

            if wrap:
                data = [data]

            # put data in history
            if snapshot:
                history.append(data)

            # put data into queue
            await asyncio.gather(
                *(asyncio.create_task(queue.put(data)) for queue in queue_map.values())
            )

            # TODO expect response from clients?
            return data

        super().__init__(foo=_req, inputs=1)
        self._name = "WebSocketServer"
        node >> self

        # set server attribute so it can be accessed
        self.set("server", server)

        # install get handler
        server.router.add_get(path, _handler)

        # Initialize application to None, might be managed outside
        self.set("app", None)
        self.set("site", None)

        if run:
            # setup runners so that we start the application
            async def _start(self=self, server=server, host=host, port=port):
                # https://docs.aiohttp.org/en/v3.0.1/web_reference.html#running-applications
                runner = web.AppRunner(server)
                self.app = runner
                await runner.setup()

                site = web.TCPSite(runner, host=host, port=port)
                self.site = site
                await site.start()

            async def _shutdown(self=self, server=server, host=host, port=port):
                await self.site.stop()
                await self.app.cleanup()

            self._onstarts = (_start,)
            self._onstops = (_shutdown,)


Node.websocket = WebSocket
Node.websocketServer = WebSocketServer
