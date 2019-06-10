import asyncio
import time
import types
from aiostream.stream import zip
from .base import _wrap, FunctionWrapper, Foo, Const


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
                await asyncio.sleep(max(0, interval-(t2-t1)))
            repeat -= 1

    return _wrap(_repeater, dict(foo=foo, repeat=repeat, interval=interval), name='Timer', wraps=(foo,), share=foo)


def Delay(f_wrap, kwargs=None, delay=1):
    if not isinstance(f_wrap, FunctionWrapper):
        f_wrap = Foo(f_wrap, kwargs or {})

    async def _delay(f_wrap, delay):
        async for f in f_wrap():
            yield f
            await asyncio.sleep(delay)

    return _wrap(_delay, dict(f_wrap=f_wrap, delay=delay), name='Delay', wraps=(f_wrap,), share=f_wrap)


def State(foo, foo_kwargs=None, **state):
    foo_kwargs = foo_kwargs or {}
    foo = _wrap(foo, foo_kwargs, name=foo.__name__, wraps=(foo,), state=state)
    return foo


def Apply(foo, f_wrap, foo_kwargs=None):
    if not isinstance(f_wrap, FunctionWrapper):
        raise Exception('Apply expects a tributary')
    foo_kwargs = foo_kwargs or {}
    foo = Foo(foo, foo_kwargs)
    foo._wraps = foo._wraps + (f_wrap, )

    async def _apply(foo):
        async for f in f_wrap():
            yield foo(f)

    return _wrap(_apply, dict(foo=foo), name='Apply', wraps=(foo,), share=foo)


def Window(foo, foo_kwargs=None, size=-1, full_only=True):
    foo_kwargs = foo_kwargs or {}
    foo = Foo(foo, foo_kwargs)

    accum = []

    async def _window(foo, size, full_only, accum):
        async for x in foo():
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


def Unroll(foo_or_val, kwargs=None):
    if not isinstance(foo_or_val, types.FunctionType):
        foo = Const(foo_or_val)
    else:
        foo = Foo(foo_or_val, kwargs or {})

    async def _unroll(foo):
        async for ret in foo():
            if isinstance(ret, list):
                for f in ret:
                    yield f
            elif isinstance(ret, types.AsyncGeneratorType):
                async for f in ret:
                    yield f

    return _wrap(_unroll, dict(foo=foo), name='Unroll', wraps=(foo,), share=foo)


def UnrollDataFrame(foo_or_val, kwargs=None, json=True, wrap=False):
    if not isinstance(foo_or_val, types.FunctionType):
        foo = Const(foo_or_val)
    else:
        foo = Foo(foo_or_val, kwargs or {})

    async def _unrolldf(foo):
        async for df in foo():
            for i in range(len(df)):
                row = df.iloc[i]
                if json:
                    data = row.to_dict()
                    data['index'] = row.name
                    yield data
                else:
                    yield row

    return _wrap(_unrolldf, dict(foo=foo), name='UnrollDataFrame', wraps=(foo,), share=foo)


def Merge(f_wrap1, f_wrap2):
    if not isinstance(f_wrap1, FunctionWrapper):
        if not isinstance(f_wrap1, types.FunctionType):
            f_wrap1 = Const(f_wrap1)
        else:
            f_wrap1 = Foo(f_wrap1)

    if not isinstance(f_wrap2, FunctionWrapper):
        if not isinstance(f_wrap2, types.FunctionType):
            f_wrap2 = Const(f_wrap2)
        else:
            f_wrap2 = Foo(f_wrap2)

    async def _merge(foo1, foo2):
        async for gen1, gen2 in zip(foo1(), foo2()):
            if isinstance(gen1, types.AsyncGeneratorType) and \
               isinstance(gen2, types.AsyncGeneratorType):
                async for f1, f2 in zip(gen1, gen2):
                    yield [f1, f2]
            elif isinstance(gen1, types.AsyncGeneratorType):
                async for f1 in gen1:
                    yield [f1, gen2]
            elif isinstance(gen2, types.AsyncGeneratorType):
                async for f2 in gen2:
                    yield [gen1, f2]
            else:
                yield [gen1, gen2]

    return _wrap(_merge, dict(foo1=f_wrap1, foo2=f_wrap2), name='Merge', wraps=(f_wrap1, f_wrap2), share=None)


def ListMerge(f_wrap1, f_wrap2):
    if not isinstance(f_wrap1, FunctionWrapper):
        if not isinstance(f_wrap1, types.FunctionType):
            f_wrap1 = Const(f_wrap1)
        else:
            f_wrap1 = Foo(f_wrap1)

    if not isinstance(f_wrap2, FunctionWrapper):
        if not isinstance(f_wrap2, types.FunctionType):
            f_wrap2 = Const(f_wrap2)
        else:
            f_wrap2 = Foo(f_wrap2)

    async def _merge(foo1, foo2):
        async for gen1, gen2 in zip(foo1(), foo2()):
            if isinstance(gen1, types.AsyncGeneratorType) and \
               isinstance(gen2, types.AsyncGeneratorType):
                async for f1, f2 in zip(gen1, gen2):
                    ret = []
                    ret.extend(f1)
                    ret.extend(f1)
                    yield ret
            elif isinstance(gen1, types.AsyncGeneratorType):
                async for f1 in gen1:
                    ret = []
                    ret.extend(f1)
                    ret.extend(gen2)
                    yield ret
            elif isinstance(gen2, types.AsyncGeneratorType):
                async for f2 in gen2:
                    ret = []
                    ret.extend(gen1)
                    ret.extend(f2)
                    yield ret
            else:
                ret = []
                ret.extend(gen1)
                ret.extend(gen2)
                yield ret

    return _wrap(_merge, dict(foo1=f_wrap1, foo2=f_wrap2), name='ListMerge', wraps=(f_wrap1, f_wrap2), share=None)


def DictMerge(f_wrap1, f_wrap2):
    if not isinstance(f_wrap1, FunctionWrapper):
        if not isinstance(f_wrap1, types.FunctionType):
            f_wrap1 = Const(f_wrap1)
        else:
            f_wrap1 = Foo(f_wrap1)

    if not isinstance(f_wrap2, FunctionWrapper):
        if not isinstance(f_wrap2, types.FunctionType):
            f_wrap2 = Const(f_wrap2)
        else:
            f_wrap2 = Foo(f_wrap2)

    async def _merge(foo1, foo2):
        async for gen1, gen2 in zip(foo1(), foo2()):
            if isinstance(gen1, types.AsyncGeneratorType) and \
               isinstance(gen2, types.AsyncGeneratorType):
                async for f1, f2 in zip(gen1, gen2):
                    ret = {}
                    ret.update(f1)
                    ret.update(f1)
                    yield ret
            elif isinstance(gen1, types.AsyncGeneratorType):
                async for f1 in gen1:
                    ret = {}
                    ret.update(f1)
                    ret.update(gen2)
                    yield ret
            elif isinstance(gen2, types.AsyncGeneratorType):
                async for f2 in gen2:
                    ret = {}
                    ret.update(gen1)
                    ret.update(f2)
                    yield ret
            else:
                ret = {}
                ret.update(gen1)
                ret.update(gen2)
                yield ret

    return _wrap(_merge, dict(foo1=f_wrap1, foo2=f_wrap2), name='DictMerge', wraps=(f_wrap1, f_wrap2), share=None)


def Reduce(*f_wraps):
    raise NotImplementedError()
