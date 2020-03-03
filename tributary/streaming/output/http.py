import requests
import json as JSON
from .output import _OUTPUT_GRAPHVIZSHAPE
from ..base import Node
from ...base import StreamEnd


def HTTP(node, url='', json=False, wrap=False, field=None, proxies=None, cookies=None):
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

    ret = Node(foo=_send, name='Http', inputs=1, graphvizshape=_OUTPUT_GRAPHVIZSHAPE)
    node._downstream.append((ret, 0))
    ret._upstream.append(node)
    return ret
