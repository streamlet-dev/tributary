import types
import math
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
                async for f2 in gen2:
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
    return unary(lambda x: -1 * x, foo, foo_kwargs, _name='Negate')


def Invert(foo, foo_kwargs=None):
    return unary(lambda x: 1 / x, foo, foo_kwargs, _name='Invert')


def Not(foo, foo_kwargs=None):
    return unary(lambda x: not x, foo, foo_kwargs, _name='Not')


def Add(foo1, foo2, foo1_kwargs=None, foo2_kwargs=None):
    return bin(lambda x, y: x + y, foo1, foo2, foo1_kwargs, foo2_kwargs, _name='Add')


def Sub(foo1, foo2, foo1_kwargs=None, foo2_kwargs=None):
    return bin(lambda x, y: x - y, foo1, foo2, foo1_kwargs, foo2_kwargs, _name='Sub')


def Mult(foo1, foo2, foo1_kwargs=None, foo2_kwargs=None):
    return bin(lambda x, y: x * y, foo1, foo2, foo1_kwargs, foo2_kwargs, _name='Mult')


def Div(foo1, foo2, foo1_kwargs=None, foo2_kwargs=None):
    return bin(lambda x, y: x / y, foo1, foo2, foo1_kwargs, foo2_kwargs, _name='Div')


def RDiv(foo1, foo2, foo1_kwargs=None, foo2_kwargs=None):
    return bin(lambda x, y: y / x, foo1, foo2, foo1_kwargs, foo2_kwargs, _name='Div')


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


def NotEqual(foo1, foo2, foo1_kwargs=None, foo2_kwargs=None):
    return bin(lambda x, y: x == y, foo1, foo2, foo1_kwargs, foo2_kwargs, _name='NotEqual')


def Lt(foo1, foo2, foo1_kwargs=None, foo2_kwargs=None):
    return bin(lambda x, y: x < y, foo1, foo2, foo1_kwargs, foo2_kwargs, _name='Less')


def Le(foo1, foo2, foo1_kwargs=None, foo2_kwargs=None):
    return bin(lambda x, y: x <= y, foo1, foo2, foo1_kwargs, foo2_kwargs, _name='Less-than-or-equal')


def Gt(foo1, foo2, foo1_kwargs=None, foo2_kwargs=None):
    return bin(lambda x, y: x > y, foo1, foo2, foo1_kwargs, foo2_kwargs, _name='Greater')


def Ge(foo1, foo2, foo1_kwargs=None, foo2_kwargs=None):
    return bin(lambda x, y: x >= y, foo1, foo2, foo1_kwargs, foo2_kwargs, _name='Greater-than-or-equal')


def Sin(foo, foo_kwargs=None):
    return unary(lambda x: math.sin(x), foo, foo_kwargs, _name='Sin')


def Cos(foo, foo_kwargs=None):
    return unary(lambda x: math.cos(x), foo, foo_kwargs, _name='Cos')


def Tan(foo, foo_kwargs=None):
    return unary(lambda x: math.tan(x), foo, foo_kwargs, _name='Tan')

# Arithmetic
FunctionWrapper.__add__ = Add
FunctionWrapper.__radd__ = Add
FunctionWrapper.__sub__ = Sub
FunctionWrapper.__rsub__ = Sub
FunctionWrapper.__mul__ = Mult
FunctionWrapper.__rmul__ = Mult
FunctionWrapper.__div__ = Div
FunctionWrapper.__rdiv__ = RDiv
FunctionWrapper.__truediv__ = Div
FunctionWrapper.__rtruediv__ = RDiv

FunctionWrapper.__pow__ = Pow
FunctionWrapper.__rpow__ = Pow
FunctionWrapper.__mod__ = Mod
FunctionWrapper.__rmod__ = Mod

# Logical
# FunctionWrapper.__and__ = And
# FunctionWrapper.__or__ = Or
# FunctionWrapper.__invert__ = Not
# TODO use __bool__ operator

# Converters
# FunctionWrapper.int = Int
# FunctionWrapper.float = Float

# Comparator
FunctionWrapper.__lt__ = Lt
FunctionWrapper.__le__ = Le
FunctionWrapper.__gt__ = Gt
FunctionWrapper.__ge__ = Ge
FunctionWrapper.__eq__ = Equal
FunctionWrapper.__ne__ = NotEqual
FunctionWrapper.__neg__ = Negate
# FunctionWrapper.__nonzero__ = Bool  # Py2 compat
# FunctionWrapper.__len__ = Len

# Numpy
# FunctionWrapper.__array_ufunc__ = __array_ufunc__

# Math
# FunctionWrapper.log = Log
FunctionWrapper.sin = Sin
FunctionWrapper.cos = Cos
FunctionWrapper.tan = Tan
# FunctionWrapper.arcsin = Arcsin
# FunctionWrapper.arccos = Arccos
# FunctionWrapper.arctan = Arctan
# FunctionWrapper.sqrt = Sqrt
# FunctionWrapper.exp = Exp
# FunctionWrapper.erf = Erf
