import json as JSON
import math
import numpy as np
from aioconsole import ainput
from ..node import Node
from ...base import StreamEnd


_INPUT_GRAPHVIZSHAPE = "box"


def _gen():
    S = 100
    T = 252
    mu = 0.25
    vol = 0.5

    returns = np.random.normal(mu / T, vol / math.sqrt(T), T) + 1
    _list = returns.cumprod() * S
    return _list


class Timer(Node):
    """Streaming wrapper to periodically call a callable `count` times
       with a delay of `interval` in between

    Arguments:
        foo (callable): callable to call
        foo_kwargs (dict): kwargs for callable
        count (int): number of times to call, 0 means infinite (or until generator is complete)
        interval (int/float): minimum delay between calls (can be more due to async scheduling)
    """

    def __init__(self, foo, foo_kwargs=None, count=1, interval=0, **kwargs):
        super().__init__(
            foo=foo,
            foo_kwargs=foo_kwargs,
            name="Timer[{}]".format(foo.__name__),
            inputs=0,
            execution_max=count,
            delay_interval=interval,
            graphvizshape=_INPUT_GRAPHVIZSHAPE,
            **kwargs
        )


class Const(Timer):
    """Streaming wrapper to return a scalar value

    Arguments:
        value (any): value to return
        count (int): number of times to call, 0 means infinite
    """

    def __init__(self, value, count=0, **kwargs):
        super().__init__(foo=lambda: value, count=count, interval=0, **kwargs)
        self._name = "Const[{}]".format(value)


class Curve(Timer):
    """Streaming wrapper to output a series of values

    Arguments:
        value (any): value to unroll and return
    """

    def __init__(self, value, **kwargs):
        def foo(curve=value):
            for v in curve:
                yield v

        super().__init__(foo=foo, count=0, **kwargs)
        self._name = "Curve[{}]".format(len(value))


class Foo(Timer):
    """Streaming wrapper to periodically call a function `count` times
       with a delay of `interval` in between

    Arguments:
        foo (callable): callable to call
        foo_kwargs (dict): kwargs for callable
        count (int): number of times to call, 0 means infinite (or until generator is complete)
        interval (int/float): minimum delay between calls (can be more due to async scheduling)
    """

    def __init__(self, foo, foo_kwargs=None, count=0, interval=0, **kwargs):
        super().__init__(
            foo=foo, foo_kwargs=foo_kwargs, count=count, interval=interval, **kwargs
        )
        self._name = "Foo[{}]".format(foo.__name__)


class Random(Foo):
    """Yield a random dictionary of data

    Args:
        count (int): number of elements to yield
        interval (float): interval to wait between yields
    """

    def __init__(self, count=10, interval=0.1, **kwargs):
        def _random(count=count, interval=interval):
            step = 0
            while step < count:
                x = {y: _gen() for y in ("A", "B", "C", "D")}
                for i in range(len(x["A"])):
                    if step >= count:
                        break
                    yield {
                        "A": x["A"][i],
                        "B": x["B"][i],
                        "C": x["C"][i],
                        "D": x["D"][i],
                    }
                    step += 1

        super().__init__(foo=_random, count=count, interval=interval, **kwargs)
        self._name = "Random"


class Console(Foo):
    """Read from console input, optionally parse as json

    Args:
        message (str): message to print for input
        json (bool): json-parse input
    """

    def __init__(self, message="", json=False, **kwargs):
        async def _input(message=message, json=json):
            try:
                val = await ainput(message)
                if json:
                    val = JSON.loads(val)
                return val
            except KeyboardInterrupt:
                return StreamEnd()
            except BaseException:
                # TODO other options?
                raise

        super().__init__(foo=_input, message=message, json=json, **kwargs)
        self._name = "Console"


class Queue(Foo):
    """Streaming wrapper to emit as values are received from an asynchronous queue

    Arguments:
        queue (Queue): asyncio Queue
    """

    def __init__(self, queue, **kwargs):
        async def foo(queue=queue):
            return await queue.get()

        super().__init__(foo=foo, count=0, **kwargs)
        self._name = "Queue"


QueueSource = Queue
