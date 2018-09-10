import time
import types
from functools import partial

real_print = print


def _wrap(foo, foo_kwargs):
    if not isinstance(foo, FunctionWrapper):
        return FunctionWrapper(foo, foo_kwargs)
    return foo


class FunctionWrapper(object):
    def __init__(self, foo, foo_kwargs):
        self.foo = foo
        self.foo_kwargs = foo_kwargs

    def __call__(self):
        real_print('calling', self.foo)
        ret = self.foo(**self.foo_kwargs)
        if isinstance(ret, types.GeneratorType):
            for r in ret:
                yield r
        else:
            yield ret


def timer(foo, kwargs=None, interval=1, repeat=0):
    kwargs = kwargs or {}

    _p = _wrap(foo, kwargs)

    def _repeater(repeat, interval):
        while repeat > 0:
            yield _p()
            repeat -= 1
            time.sleep(interval)

    return _wrap(_repeater, dict(repeat=repeat, interval=interval))


def add(foo1, foo2, foo1_kwargs=None, foo2_kwargs=None):
    foo1_kwargs = None or {}
    foo2_kwargs = None or {}
    foo1 = _wrap(foo1, foo1_kwargs)
    foo2 = _wrap(foo2, foo2_kwargs)

    def _add(foo1, foo2):
        for gen1, gen2 in zip(foo1(), foo2()):
            for f1, f2 in zip(gen1, gen2):
                yield f1 + f2

    return _wrap(_add, dict(foo1=foo1, foo2=foo2))


def print(foo, foo_kwargs=None):
    foo_kwargs = foo_kwargs or {}
    foo = _wrap(foo, foo_kwargs)

    def _print(foo):
        for r in foo():
            real_print(r)

    return _wrap(_print, dict(foo=foo))


def run(foo, **kwargs):
    foo = _wrap(foo, kwargs)
    for item in foo():
        pass
