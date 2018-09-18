import types
from aiostream.stream import zip
from ..base import _wrap, FunctionWrapper


def unary(lam, foo, foo_kwargs=None, _name=''):
    foo_kwargs = None or {}
    foo = _wrap(foo, foo_kwargs)

    async def _unary(foo):
        async for gen in foo():
            if isinstance(gen, types.AsyncGeneratorType):
                async for f in gen:
                    if isinstance(f, types.CoroutineType):
                        yield lam(await f)
                    else:
                        yield lam(f)
            elif isinstance(gen, types.CoroutineType):
                yield lam(await gen)
            else:
                yield lam(gen)

    return _wrap(_unary, dict(foo=foo), name=_name or 'Unary', wraps=(foo,), share=None)


def bin(lam, foo1, foo2, foo1_kwargs=None, foo2_kwargs=None, _name=''):
    foo1_kwargs = None or {}
    foo2_kwargs = None or {}
    foo1 = _wrap(foo1, foo1_kwargs)
    foo2 = _wrap(foo2, foo2_kwargs)

    async def _bin(foo1, foo2):
        # TODO replace with merge
        async for gen1, gen2 in zip(foo1(), foo2()):
            if isinstance(gen1, types.AsyncGeneratorType) and isinstance(gen2, types.AsyncGeneratorType):
                print(1)
                async for f1, f2 in zip(gen1, gen2):
                    if isinstance(f1, types.CoroutineType):
                        f1 = await f1
                    if isinstance(f2, types.CoroutineType):
                        f2 = await f2
                    yield lam(f1, f2)
            elif isinstance(gen1, types.AsyncGeneratorType):
                async for f1 in gen1:
                    if isinstance(f1, types.CoroutineType):
                        f1 = await f1
                    if isinstance(gen2, types.CoroutineType):
                        gen2 = await gen2
                    yield lam(f1, gen2)
            elif isinstance(gen2, types.AsyncGeneratorType):
                print(3)
                async for f2 in gen2:
                    print(31)
                    if isinstance(gen1, types.CoroutineType):
                        gen1 = await gen1
                    if isinstance(f2, types.CoroutineType):
                        f2 = await f2

                    yield lam(gen1, f2)
            else:
                if isinstance(gen1, types.CoroutineType):
                    gen1 = await gen1
                if isinstance(gen2, types.CoroutineType):
                    gen2 = await gen2
                yield lam(gen1, gen2)

    return _wrap(_bin, dict(foo1=foo1, foo2=foo2), name=_name or 'Binary', wraps=(foo1, foo2), share=None)


def Noop(foo, foo_kwargs=None):
    return unary(lambda x: x, foo, foo_kwargs, _name='Noop')


def Negate(foo, foo_kwargs=None):
    return unary(lambda x: -1*x, foo, foo_kwargs, _name='Negate')


def Invert(foo, foo_kwargs=None):
    return unary(lambda x: 1/x, foo, foo_kwargs, _name='Invert')


def Not(foo, foo_kwargs=None):
    return unary(lambda x: not x, foo, foo_kwargs, _name='Not')


def Add(foo1, foo2, foo1_kwargs=None, foo2_kwargs=None):
    return bin(lambda x, y: x+y, foo1, foo2, foo1_kwargs, foo2_kwargs, _name='Add')


def Sub(foo1, foo2, foo1_kwargs=None, foo2_kwargs=None):
    return bin(lambda x, y: x-y, foo1, foo2, foo1_kwargs, foo2_kwargs, _name='Sub')


def Mult(foo1, foo2, foo1_kwargs=None, foo2_kwargs=None):
    return bin(lambda x, y: x*y, foo1, foo2, foo1_kwargs, foo2_kwargs, _name='Mult')


def Div(foo1, foo2, foo1_kwargs=None, foo2_kwargs=None):
    return bin(lambda x, y: x/y, foo1, foo2, foo1_kwargs, foo2_kwargs, _name='Div')


def Mod(foo1, foo2, foo1_kwargs=None, foo2_kwargs=None):
    return bin(lambda x, y: x % y, foo1, foo2, foo1_kwargs, foo2_kwargs, _name='Mod')


def Pow(foo1, foo2, foo1_kwargs=None, foo2_kwargs=None):
    return bin(lambda x, y: x**y, foo1, foo2, foo1_kwargs, foo2_kwargs, _name='Pow')


def And(foo1, foo2, foo1_kwargs=None, foo2_kwargs=None):
    return bin(lambda x, y: x and y, foo1, foo2, foo1_kwargs, foo2_kwargs, _name='And')


def Or(foo1, foo2, foo1_kwargs=None, foo2_kwargs=None):
    return bin(lambda x, y: x or y, foo1, foo2, foo1_kwargs, foo2_kwargs, _name='Or')


def Equal(foo1, foo2, foo1_kwargs=None, foo2_kwargs=None):
    return bin(lambda x, y: x == y, foo1, foo2, foo1_kwargs, foo2_kwargs, _name='Equal')


def Less(foo1, foo2, foo1_kwargs=None, foo2_kwargs=None):
    return bin(lambda x, y: x < y, foo1, foo2, foo1_kwargs, foo2_kwargs, _name='Less')


def More(foo1, foo2, foo1_kwargs=None, foo2_kwargs=None):
    return bin(lambda x, y: x > y, foo1, foo2, foo1_kwargs, foo2_kwargs, _name='More')


# Arithmetic
FunctionWrapper.__add__ = Add
FunctionWrapper.__sub__ = Sub
FunctionWrapper.__mul__ = Mult
FunctionWrapper.__div__ = Div
FunctionWrapper.__truediv__ = Div
FunctionWrapper.__pow__ = Pow
FunctionWrapper.__mod__ = Mod

# Logical
# FunctionWrapper.__and__ = And
# FunctionWrapper.__or__ = Or
# FunctionWrapper.__invert__ = Not
# TODO use __bool__ operator

# Comparator
FunctionWrapper.__lt__ = Less
FunctionWrapper.__gt__ = More
