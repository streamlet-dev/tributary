import time
from functools import partial


def timer(foo, args=None, kwargs=None, interval=1, repeat=0):
    args = args or ()
    kwargs = kwargs or {}

    _p = partial(foo, *args, **kwargs)

    def _repeater(repeat=repeat, *args, **kwargs):
        while repeat > 0:
            yield _p(*args, **kwargs)
            repeat -= 1
            time.sleep(interval)

    return _repeater


def add(foo1, foo2, foo1_kwargs, foo2_kwargs):
    return foo1(**foo1_kwargs) + foo2(**foo2_kwargs)
