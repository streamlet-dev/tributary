import aiohttp
import asyncio
import json as JSON
from aiohttp import web
from .input import Foo
from ...base import TributaryException


class HTTP(Foo):
    """Connect to url and yield results

    Args:
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
        url,
        interval=1,
        repeat=1,
        json=False,
        wrap=False,
        field=None,
        proxies=None,
        cookies=None,
        response_handler=None,
    ):
        async def _req(
            url=url,
            interval=interval,
            repeat=repeat,
            json=json,
            wrap=wrap,
            field=field,
            proxies=proxies,
            cookies=cookies,
        ):
            count = 0 if repeat >= 0 else float("-inf")  # make less than anything
            while count < repeat:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        url, cookies=cookies, proxy=proxies
                    ) as response:

                        if response_handler and callable(response_handler):
                            yield response_handler(response)

                        else:
                            msg = await response.text()

                            if msg is None or response.status != 200:
                                break

                            if json:
                                msg = JSON.loads(msg)

                            if field:
                                msg = msg[field]

                            if wrap:
                                msg = [msg]

                            yield msg

                        if interval:
                            await asyncio.sleep(interval)

                        if repeat >= 0:
                            count += 1

        super().__init__(foo=_req)
        self._name = "HTTP"


class HTTPServer(Foo):
    """Host a server and yield posted data

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
        path="/",
        json=False,
        wrap=False,
        field=None,
        server=None,
        meth="post",
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
        async def _handler(request, queue=queue, response_handler=response_handler):
            # put the request into the queue
            await queue.put(request)

            # if custom response handler is given, use that to determine response
            if response_handler and callable(response_handler):
                return await response_handler(request)
            elif response_handler and isinstance(response_handler, str):
                return web.Response(text=response_handler)
            else:
                # just put an empty ok
                return web.Response()

        # tributary node handler
        async def _req(
            json=json,
            wrap=wrap,
            field=field,
            queue=queue,
            request_handler=request_handler,
        ):
            # get request from queue
            request = await queue.get()

            # if custom request processor, use that
            if request_handler and callable(request_handler):
                data = await request_handler(request)
            else:
                # use text if not json
                if not json:
                    data = await request.text()
                else:
                    # read body as json
                    data = await request.json()

            # if specifying a field
            if field:
                data = data[field]

            # if wrapping into list
            if wrap:
                data = [data]

            # return the end result
            return data

        super().__init__(foo=_req)
        self._name = "HTTPServer"

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
                await self.runner.cleanup()

            self._onstarts = (_start,)
            self._onstops = (_shutdown,)
