import time
import math
import numpy as np

from ...thread import run
from ...base import StreamNone, StreamEnd
from ..base import _wrap, FunctionWrapper
from .ws import WebSocket, SyncWebSocket
from .http import HTTP, SyncHTTP
from .socketio import SocketIO, SyncSocketIO
from .kafka import Kafka, SyncKafka


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


def Functional(foo, foo_kwargs, f_wrap=None, callback_name=None, on_data_name=None):
    if f_wrap is not None:
        if not isinstance(f_wrap, FunctionWrapper):
            raise Exception('Apply expects a tributary')

    foo = _wrap(foo, foo_kwargs or {}, name='Foo', wraps=(foo,))

    def _foo(foo, callback_name, f_wrap):
        if f_wrap:
            for x in f_wrap():
                if callback_name:
                    foo = partial(foo, {callback_name: lambda x: _data.append(x)})
                for y in run(foo):
            if isinstance(x, StreamNone):
                continue
            elif not x or isinstance(x, StreamEnd):
                break
            yield x

    return _wrap(_foo, dict(foo=foo), name='Functional', wraps=(foo,))
