import requests
import json as JSON
from ..base import Node
from ...base import StreamEnd


class HTTP(Node):
    '''Connect to url and post results to it

    Args:
        foo (callable): input stream
        foo_kwargs (dict): kwargs for the input stream
        url (str): url to post to
        json (bool): dump data as json
        wrap (bool): wrap input in a list
        field (str): field to index result by
        proxies (list): list of URL proxies to pass to requests.post
        cookies (list): list of cookies to pass to requests.post
    '''

    def __init__(self, node, url='', json=False, wrap=False, field=None, proxies=None, cookies=None):
        def _send(data, url=url, json=json, wrap=wrap, field=field, proxies=proxies, cookies=cookies):
            if wrap:
                data = [data]
            if json:
                data = JSON.dumps(data)

            msg = requests.post(url, data=data, cookies=cookies, proxies=proxies)

            if msg is None:
                return StreamEnd()

            if msg.status_code != 200:
                return msg

            if json:
                msg = msg.json()

            if field:
                msg = msg[field]

            if wrap:
                msg = [msg]

            return msg

        super().__init__(foo=_send, name='Http', inputs=1)
        node._downstream.append((self, 0))
        self._upstream.append(node)
