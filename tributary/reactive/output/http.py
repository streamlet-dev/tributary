import requests
import ujson
from ..base import _wrap


def HTTP(foo, foo_kwargs=None, *args, **kwargs):
    return AsyncHTTP(foo, foo_kwargs, *args, **kwargs)


def AsyncHTTP(foo, foo_kwargs=None, url='', json=False, wrap=False, field=None, proxies=None, cookies=None):
    foo_kwargs = foo_kwargs or {}
    foo = _wrap(foo, foo_kwargs)

    def _send(foo, url, json=False, wrap=False, field=None, proxies=None, cookies=None):
        for data in foo():
            if wrap:
                data = [data]
            if json:
                data = ujson.dumps(data)
            print('*'*10)
            print(url)
            msg = requests.post(url, data=data, cookies=cookies, proxies=proxies)
            print(msg)
            print('*'*10)

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

    return _wrap(_send, dict(foo=foo, url=url, json=json, wrap=wrap, field=field, proxies=proxies, cookies=cookies), name='HTTP')
