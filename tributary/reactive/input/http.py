import aiohttp
import json as JSON
import time
import functools
from ..base import _wrap


def AsyncHTTP(url, interval=1, repeat=1, json=False, wrap=False, field=None, proxies=None, cookies=None):
    '''Connect to url and yield results

    Args:
        url (str): url to connect to
        interval (int): interval to re-query
        repeat (int): number of times to request
        json (bool): load http content data as json
        wrap (bool): wrap result in a list
        field (str): field to index result by
        proxies (list): list of URL proxies to pass to requests.get
        cookies (list): list of cookies to pass to requests.get
    '''
    async def _req(url, interval=1, repeat=1, json=False, wrap=False, field=None, proxies=None, cookies=None):
        count = 0
        while count < repeat:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, cookies=cookies, proxy=proxies) as response:
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
                        time.sleep(interval)
                    if repeat >= 0:
                        count += 1

    return _wrap(_req, dict(url=url, interval=interval, repeat=repeat, json=json, wrap=wrap, field=field, proxies=proxies, cookies=cookies), name='HTTP')


@functools.wraps(AsyncHTTP)
def HTTP(url, *args, **kwargs):
    return AsyncHTTP(url, *args, **kwargs)
