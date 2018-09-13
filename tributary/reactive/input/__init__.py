import time
import math
import numpy as np
from ..base import _wrap
from tornado.websocket import websocket_connect


def _gen():
    S = 100
    T = 252
    mu = 0.25
    vol = 0.5

    returns = np.random.normal(mu / T, vol / math.sqrt(T), T) + 1
    _list = returns.cumprod() * S
    return _list


def Random(size=10, interval=0.1):

    def _random(size, interval):
        step = 0
        while step < size:
            x = {y: _gen() for y in ('A', 'B', 'C', 'D')}
            for i in range(len(x['A'])):
                if step >= size:
                    break
                yield {'A': x['A'][i],
                       'B': x['B'][i],
                       'C': x['C'][i],
                       'D': x['D'][i]}
                time.sleep(interval)
                step += 1

    return _wrap(_random, dict(size=size, interval=interval), name='Random')


def WebSocket(url):
    def _listen(url):
        conn = yield websocket_connect(url)
        while True:
            msg = yield conn.read_message()
            if msg is None:
                break
            # Do something with msg

    return _wrap(_listen, dict(url=url), name='WebSocket')

