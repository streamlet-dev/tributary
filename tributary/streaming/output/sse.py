import asyncio
from collections import deque
import json as JSON
from aiohttp import web
from aiohttp_sse import sse_response
from .output import Foo
from ..node import Node


class SSE(Foo):
    """Host an sse server and send results on requests

    Args:
        path (str): route on which to host sse server
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
        ):
            async with sse_response(request) as resp:
                if resp not in queue_map:
                    # create a queue for this client
                    queue_map[resp] = asyncio.Queue()

                    # send history if snapshotting
                    for data in history:
                        if response_handler and callable(response_handler):
                            data = await response_handler(request, data)
                        await resp.send(data)

                queue = queue_map[resp]

                try:
                    while not resp.task.done():
                        # put the request into the queue
                        data = await queue.get()

                        # TODO move this?
                        if request_handler and callable(request_handler):
                            data = await request_handler(request)

                        # if custom response handler is given, use that to determine response
                        if response_handler and callable(response_handler):
                            data = await response_handler(request, data)
                            await resp.send(data)

                        elif response_handler and isinstance(response_handler, str):
                            await resp.send(response_handler)
                        else:
                            # just put an ok with data
                            await resp.send(JSON.dumps(data))
                finally:
                    # remove from queue
                    queue_map.pop(resp)

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
                *(asyncio.create_task(queue.put(data) for queue in queue_map.values()))
            )

            # TODO expect response from clients?
            return data

        super().__init__(foo=_req, inputs=1)
        self._name = "SSE"
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


Node.sse = SSE
