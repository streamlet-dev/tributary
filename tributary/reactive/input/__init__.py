import asyncio
import math
import numpy as np

from ..base import _wrap
from .file import File  # noqa: F401
from .http import HTTP, AsyncHTTP  # noqa: F401
from .kafka import Kafka, AsyncKafka  # noqa: F401
from .socketio import SocketIO, AsyncSocketIO  # noqa: F401
from .ws import WebSocket, AsyncWebSocket  # noqa: F401


def _gen():
    S = 100
    T = 252
    mu = 0.25
    vol = 0.5

    returns = np.random.normal(mu / T, vol / math.sqrt(T), T) + 1
    _list = returns.cumprod() * S
    return _list


def Random(size=10, interval=0.1):

    async def _random(size, interval):
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
                await asyncio.sleep(interval)
                step += 1

    return _wrap(_random, dict(size=size, interval=interval), name='Random')
