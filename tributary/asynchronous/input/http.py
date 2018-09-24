import time
import requests
from ..base import _wrap


def HTTP(url, *args, **kwargs):
    return AsyncHTTP(url, *args, **kwargs)


def AsyncHTTP(url, interval=1, repeat=1, json=False, wrap=False, field=None, proxies=None, cookies=None):
    async def _req(url, interval=1, repeat=1, json=False, wrap=False, field=None, proxies=None, cookies=None):
        count = 0
        while count < repeat:
            msg = requests.get(url, cookies=cookies, proxies=proxies)

            if msg is None or msg.status_code != 200:
                break

            if json:
                msg = msg.json()

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
