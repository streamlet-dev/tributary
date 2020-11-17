import json as JSON
from aiohttp_sse_client import client as sse_client
from .input import Foo


class SSE(Foo):
    """Connect to SSE url and yield results

    Args:
        url (str): url to connect to
        json (bool): load http content data as json
        wrap (bool): wrap result in a list
        field (str): field to index result by
    """

    def __init__(self, url, json=False, wrap=False, field=None):
        async def _req(url=url, json=json, wrap=wrap, field=field):
            async with sse_client.EventSource(url) as event_source:
                async for event in event_source:
                    data = event.data

                    if json:
                        data = JSON.loads(data)
                    if field:
                        data = data[field]
                    if wrap:
                        data = [data]
                    yield data

        super().__init__(foo=_req)
        self._name = "SSE"
