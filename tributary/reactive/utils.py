import time
import types
from .base import _wrap, FunctionWrapper


def Const(val):
    def _always(val):
        yield val

    return _wrap(_always, dict(val=val), name='Const', wraps=(val,))


def Foo(foo, foo_kwargs=None):
    return _wrap(foo, foo_kwargs or {}, name='Foo', wraps=(foo,))


def Timer(foo_or_val, kwargs=None, interval=1, repeat=0):
    kwargs = kwargs or {}

    if not isinstance(foo_or_val, types.FunctionType):
        foo = Const(foo_or_val)
    else:
        foo = Foo(foo_or_val, kwargs)

    def _repeater(foo, repeat, interval):
        while repeat > 0:
            t1 = time.time()
            yield foo()
            t2 = time.time()

            if interval > 0:
                # sleep for rest of time that _p didnt take
                time.sleep(max(0, interval-(t2-t1)))
            repeat -= 1

    return _wrap(_repeater, dict(foo=foo, repeat=repeat, interval=interval), name='Timer', wraps=(foo,), share=foo)


def Share(f_wrap):
    if not isinstance(f_wrap, FunctionWrapper):
        raise Exception('Share expects tributary')
    f_wrap.inc()
    return f_wrap


def State(foo, foo_kwargs=None, **state):
    foo_kwargs = foo_kwargs or {}
    foo = _wrap(foo, foo_kwargs, name=foo.__name__, wraps=(foo,), state=state)
    return foo


def Apply(foo, f_wrap, foo_kwargs=None):
    if not isinstance(f_wrap, FunctionWrapper):
        raise Exception('Node expects tributary')
    foo_kwargs = foo_kwargs or {}
    foo = Foo(foo, foo_kwargs)
    foo._wraps = foo._wraps + (f_wrap, )

    def _apply(foo):
        for f in f_wrap():
            yield foo(f)

    return _wrap(_apply, dict(foo=foo), name='Apply', wraps=(foo,), share=foo)


def Window(foo, foo_kwargs=None, size=-1, full_only=True):
    foo_kwargs = foo_kwargs or {}
    foo = Foo(foo, foo_kwargs)

    accum = []

    def _window(foo, size, full_only, accum):
        for x in foo():
            if size == 0:
                yield x
            else:
                accum.append(x)

                if size > 0:
                    accum = accum[-size:]

                if full_only:
                    if len(accum) == size or size == -1:
                        yield accum
                else:
                    yield accum

    return _wrap(_window, dict(foo=foo, size=size, full_only=full_only, accum=accum), name='Window', wraps=(foo,), share=foo)


def Unroll(foo, foo_kwargs):
    foo_kwargs = foo_kwargs or {}
    foo = Foo(foo, foo_kwargs)

    def _unroll(foo):
        ret = foo()
        if isinstance(ret, list) or isinstance(ret, types.GeneratorType):
            for f in ret:
                yield f

    return _wrap(_unroll, dict(foo=foo), name='Apply', wraps=(foo,), share=foo)
