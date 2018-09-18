import types
from ..base import _wrap


def Count(foo, foo_kwargs=None):
    foo_kwargs = foo_kwargs or {}
    foo = _wrap(foo, foo_kwargs)

    def _count(foo):
        count = 0
        for gen in foo():
            if isinstance(gen, types.GeneratorType):
                for f in gen:
                    count += 1
                    yield count
            else:
                count += 1
                yield count

    return _wrap(_count, dict(foo=foo), name='Count', wraps=(foo,), share=foo)


def Sum(foo, foo_kwargs=None):
    foo_kwargs = foo_kwargs or {}
    foo = _wrap(foo, foo_kwargs)

    def _sum(foo):
        sum = 0
        for gen in foo():
            if isinstance(gen, types.GeneratorType):
                for f in gen:
                    sum += f
                    yield sum
            else:
                sum += gen
                yield sum

    return _wrap(_sum, dict(foo=foo), name='Sum', wraps=(foo,), share=foo)
