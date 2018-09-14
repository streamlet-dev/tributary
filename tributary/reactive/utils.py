import time
import types
from .base import _wrap, FunctionWrapper, Foo, Const


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

    return _wrap(_unroll, dict(foo=foo), name='Unroll', wraps=(foo,), share=foo)


def UnrollDataFrame(df, json=True, wrap=False):
    def _unrolldf(val):
        for i in range(len(df)):
            row = df.iloc[i]
            if json:
                data = row.to_dict()
                data['index'] = row.name
                yield data
            else:
                yield row

    return _wrap(_unrolldf, dict(val=df), name='UnrollDataFrame', wraps=(df,))


def Merge(f_wrap1, f_wrap2):
    if not isinstance(f_wrap1, FunctionWrapper):
        raise Exception('Merge expects a tributary')

    if not isinstance(f_wrap2, FunctionWrapper):
        raise Exception('Merge expects a tributary')

    def _merge(foo1, foo2):
        for gen1, gen2 in zip(foo1(), foo2()):
            if isinstance(gen1, types.GeneratorType) and \
               isinstance(gen2, types.GeneratorType):
                for f1, f2 in zip(gen1, gen2):
                    yield [f1, f2]
            elif isinstance(gen1, types.GeneratorType):
                for f1 in gen1:
                    yield [f1, gen2]
            elif isinstance(gen2, types.GeneratorType):
                for f2 in gen2:
                    yield [gen1, f2]
            else:
                yield [gen1, gen2]

    return _wrap(_merge, dict(foo1=f_wrap1, foo2=f_wrap2), name='Merge', wraps=(f_wrap1, f_wrap2), share=None)


def ListMerge(f_wrap1, f_wrap2):
    if not isinstance(f_wrap1, FunctionWrapper):
        raise Exception('Merge expects a tributary')

    if not isinstance(f_wrap2, FunctionWrapper):
        raise Exception('Merge expects a tributary')

    def _merge(foo1, foo2):
        for gen1, gen2 in zip(foo1(), foo2()):
            if isinstance(gen1, types.GeneratorType) and \
               isinstance(gen2, types.GeneratorType):
                for f1, f2 in zip(gen1, gen2):
                    ret = []
                    ret.extend(f1)
                    ret.extend(f1)
                    yield ret
            elif isinstance(gen1, types.GeneratorType):
                for f1 in gen1:
                    ret = []
                    ret.extend(f1)
                    ret.extend(gen2)
                    yield ret
            elif isinstance(gen2, types.GeneratorType):
                for f2 in gen2:
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
        raise Exception('Merge expects a tributary')

    if not isinstance(f_wrap2, FunctionWrapper):
        raise Exception('Merge expects a tributary')

    def _merge(foo1, foo2):
        for gen1, gen2 in zip(foo1(), foo2()):
            if isinstance(gen1, types.GeneratorType) and \
               isinstance(gen2, types.GeneratorType):
                for f1, f2 in zip(gen1, gen2):
                    ret = {}
                    ret.update(f1)
                    ret.update(f1)
                    yield ret
            elif isinstance(gen1, types.GeneratorType):
                for f1 in gen1:
                    ret = {}
                    ret.update(f1)
                    ret.update(gen2)
                    yield ret
            elif isinstance(gen2, types.GeneratorType):
                for f2 in gen2:
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


def Delay(f_wrap, delay=1):
    if not isinstance(f_wrap, FunctionWrapper):
        raise Exception('Delay expects a tributary')

    def _delay(f_wrap, delay):
        for f in f_wrap():
            yield f
            time.sleep(delay)

    return _wrap(_delay, dict(f_wrap=f_wrap, delay=delay), name='Delay', wraps=(f_wrap,), share=f_wrap)
