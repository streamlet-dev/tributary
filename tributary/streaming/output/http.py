import asyncio
import aiohttp
import json as JSON
from aiohttp import web
from .output import Foo
from ..node import Node
from ...base import StreamEnd, TributaryException


class HTTP(Foo):
    """Connect to url and post results to it

    Args:
        node (Node): input tributary
        url (str): url to connect to
        interval (int): interval to re-query
        repeat (int): number of times to request
        json (bool): load http content data as json
        wrap (bool): wrap result in a list
        field (str): field to index result by
        proxies (list): list of URL proxies to pass to requests.get
        cookies (list): list of cookies to pass to requests.get
        response_handler (Optional[callable]): custom handler to manage the response from the server
    """

    def __init__(
        self,
        node,
        url,
        json=False,
        wrap=False,
        field=None,
        proxies=None,
        cookies=None,
        response_handler=None,
    ):
        async def _send(
            data,
            url=url,
            json=json,
            wrap=wrap,
            field=field,
            proxies=proxies,
            cookies=cookies,
        ):
            if json:
                data = JSON.dumps(data)

            if field:
                data = data[field]

            if wrap:
                data = [data]

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, cookies=cookies, proxy=proxies, data=data
                ) as response:
                    if response_handler and callable(response_handler):
                        return response_handler(response)

                    msg = await response.text()

                    if msg is None:
                        return StreamEnd()
                    if response.status != 200:
                        return msg

                    if json:
                        msg = JSON.loads(msg)

                    if field:
                        msg = msg[field]

                    if wrap:
                        msg = [msg]

                    return msg

        super().__init__(foo=_send, inputs=1)
        self._name = "HTTP"
        node >> self


class HTTPServer(Foo):
    """Host a server and send results on get requests

    Args:
        path (str): route on which to host http server
        json (bool): load http content data as json
        wrap (bool): wrap result in a list
        field (str): field to index result by
        server (Optional[aiohttp.web.Application]): aiohttp application to install route
        meth (Optional[str]): string representation of method to support, in (get, post, both)
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
        server=None,
        meth="get",
        run=True,
        host="127.0.0.1",
        port=8080,
        request_handler=None,
        response_handler=None,
    ):
        # Handle method type
        if meth.lower() not in ("get", "post", "both"):
            raise TributaryException("`meth` must be in ('get', 'post', 'both')")

        # instantiate server if not existing
        server = server or web.Application()

        # create asyncio queue to manage between the http server and the tributary foo
        queue = asyncio.Queue()

        # http server handler
        async def _handler(
            request,
            queue=queue,
            request_handler=request_handler,
            response_handler=response_handler,
        ):
            if queue.empty():
                return web.Response()

            datas = []
            while not queue.empty():
                # put the request into the queue
                data = await queue.get()

                if request_handler and callable(request_handler):
                    data = await request_handler(request)
                datas.append(data)

            # if custom response handler is given, use that to determine response
            if response_handler and callable(response_handler):
                return await response_handler(request, datas)

            elif response_handler and isinstance(response_handler, str):
                return web.Response(text=response_handler)

            else:
                # just put an ok with data
                return web.Response(text=JSON.dumps(datas))

        # tributary node handler
        async def _req(data, json=json, wrap=wrap, field=field, queue=queue):
            if json:
                data = JSON.dumps(data)

            if field:
                data = data[field]

            if wrap:
                data = [data]

            # put data into queue
            await queue.put(data)

            # TODO expect response from clients?
            return data

        super().__init__(foo=_req, inputs=1)
        self._name = "HTTPServer"
        node >> self

        # set server attribute so it can be accessed
        self.set("server", server)

        # install get handler
        if meth.lower() in ("get", "both"):
            server.router.add_get(path, _handler)

        # install post handler
        if meth.lower() in ("post", "both"):
            server.router.add_post(path, _handler)

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
                return

            self._onstarts = (_start,)
            self._onstops = (_shutdown,)


Node.http = HTTP
Node.httpServer = HTTPServer
