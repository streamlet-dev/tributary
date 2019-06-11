import time
import requests
from ..base import _wrap


def HTTP(url, *args, **kwargs):
    return SyncHTTP(url, *args, **kwargs)


def SyncHTTP(url, interval=1, repeat=1, json=False, wrap=False, field=None, proxies=None, cookies=None):
    def _req(url, interval=1, repeat=1, json=False, wrap=False, field=None, proxies=None, cookies=None):
        count = 0
        while count < repeat:
            with requests.Session() as s:
                msg = s.get(url, cookies=cookies, proxies=proxies)

                if msg is None:
                    break

                if msg.status_code != 200:
                    yield msg
                    continue

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
