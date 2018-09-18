import asyncio
import time
import types
from .base import _wrap, Foo, Const


def Timer(foo_or_val, kwargs=None, interval=1, repeat=0):
    kwargs = kwargs or {}

    if not isinstance(foo_or_val, types.FunctionType):
        foo = Const(foo_or_val)
    else:
        foo = Foo(foo_or_val, kwargs)

    async def _repeater(foo, repeat, interval):
        while repeat > 0:
            t1 = time.time()
            f = foo()
            yield f
            t2 = time.time()

            if interval > 0:
                # sleep for rest of time that _p didnt take
                asyncio.sleep(max(0, interval-(t2-t1)))
            repeat -= 1

    return _wrap(_repeater, dict(foo=foo, repeat=repeat, interval=interval), name='Timer', wraps=(foo,), share=foo)
