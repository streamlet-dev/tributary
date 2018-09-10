import time
import types
from .base import _wrap, FunctionWrapper


def Const(val):
    def _always(val):
        yield val
    return _wrap(_always, dict(val=val))


def Timer(foo_or_val, kwargs=None, interval=1, repeat=0):
    kwargs = kwargs or {}

    if not isinstance(foo_or_val, types.FunctionType):
        _p = Const(foo_or_val)
    else:
        _p = _wrap(foo_or_val, kwargs)

    def _repeater(foo, repeat, interval):
        while repeat > 0:
            t1 = time.time()
            yield foo()
            t2 = time.time()

            if interval > 0:
                # sleep for rest of time that _p didnt take
                print('sleep', t2-t1-interval)
                time.sleep(max(0, t2-t1-interval))
            repeat -= 1

    return _wrap(_repeater, dict(foo=_p, repeat=repeat, interval=interval))


def Print(foo, foo_kwargs=None):
    foo_kwargs = foo_kwargs or {}
    foo = _wrap(foo, foo_kwargs)

    def _print(foo):
        for r in foo():
            print(r)

    return _wrap(_print, dict(foo=foo))


def Share(f_wrap):
    if not isinstance(f_wrap, FunctionWrapper):
        raise Exception('Share expects tributary')
    f_wrap.inc()
    return f_wrap
