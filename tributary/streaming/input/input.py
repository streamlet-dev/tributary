import asyncio
import math
import numpy as np

from ..base import Node
from ...base import StreamEnd


def _gen():
    S = 100
    T = 252
    mu = 0.25
    vol = 0.5

    returns = np.random.normal(mu / T, vol / math.sqrt(T), T) + 1
    _list = returns.cumprod() * S
    return _list


class Timer(Node):
    def __init__(self, foo, foo_kwargs=None, count=1, interval=0):
        self._count = count
        self._executed = 0
        self._interval = interval

        super().__init__(foo=foo, foo_kwargs=foo_kwargs, name='Timer[{}]'.format(foo.__name__), inputs=0)

    async def _execute(self):
        self._executed += 1
        await super()._execute()

    async def __call__(self):
        # sleep if needed
        if self._interval:
            await asyncio.sleep(self._interval)

        if self._count > 0 and self._executed >= self._count:
            self._foo = lambda: StreamEnd()

        return await self._execute()


class Const(Timer):
    def __init__(self, value, count=0):
        super().__init__(foo=lambda: value, count=count, interval=0)
        self._name = 'Const[{}]'.format(value)


class Foo(Timer):
    def __init__(self, foo, foo_kwargs=None, count=0, interval=0):
        super().__init__(foo=foo, foo_kwargs=foo_kwargs, count=count, interval=interval)
        self._name = 'Foo[{}]'.format(foo.__name__)


class Random(Foo):
    '''Yield a random dictionary of data

    Args:
        count (int): number of elements to yield
        interval (float): interval to wait between yields
    '''

    def __init__(self, count=10, interval=0.1):
        def _random(count=count, interval=interval):
            step = 0
            while step < count:
                x = {y: _gen() for y in ('A', 'B', 'C', 'D')}
                for i in range(len(x['A'])):
                    if step >= count:
                        break
                    yield {'A': x['A'][i],
                           'B': x['B'][i],
                           'C': x['C'][i],
                           'D': x['D'][i]}
                    step += 1
        super().__init__(foo=_random, count=count, interval=interval)
        self._name = 'Random'
