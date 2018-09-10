import types
from .base import _wrap


def unary(lam, foo, foo_kwargs=None):
    foo_kwargs = None or {}
    foo = _wrap(foo, foo_kwargs)

    def _unary(foo):
        for gen in foo():
            if isinstance(gen, types.GeneratorType):
                for f in gen:
                    yield lam(f)
            else:
                yield lam(gen)

    return _wrap(_unary, dict(foo=foo))


def bin(lam, foo1, foo2, foo1_kwargs=None, foo2_kwargs=None):
    foo1_kwargs = None or {}
    foo2_kwargs = None or {}
    foo1 = _wrap(foo1, foo1_kwargs)
    foo2 = _wrap(foo2, foo2_kwargs)

    def _bin(foo1, foo2):
        for gen1, gen2 in zip(foo1(), foo2()):
            if isinstance(gen1, types.GeneratorType) and \
               isinstance(gen2, types.GeneratorType):
                for f1, f2 in zip(gen1, gen2):
                    # print('ff', f1, f2)
                    yield lam(f1, f2)
            elif isinstance(gen1, types.GeneratorType):
                for f1 in gen1:
                    # print('fg', f1, gen2)
                    yield lam(f1, gen2)
            elif isinstance(gen2, types.GeneratorType):
                for f2 in gen2:
                    # print('gf', gen1, f2)
                    yield lam(gen1, f2)
            else:
                # print('gg', gen1, gen2)
                yield lam(gen1, gen2)

    return _wrap(_bin, dict(foo1=foo1, foo2=foo2))


def Noop(foo, foo_kwargs=None):
    return unary(lambda x: x, foo, foo_kwargs)


def Negate(foo, foo_kwargs=None):
    return unary(lambda x: -1*x, foo, foo_kwargs)


def Invert(foo, foo_kwargs=None):
    return unary(lambda x: 1/x, foo, foo_kwargs)


def Not(foo, foo_kwargs=None):
    return unary(lambda x: not x, foo, foo_kwargs)


def Add(foo1, foo2, foo1_kwargs=None, foo2_kwargs=None):
    return bin(lambda x, y: x+y, foo1, foo2, foo1_kwargs, foo2_kwargs)


def Sub(foo1, foo2, foo1_kwargs=None, foo2_kwargs=None):
    return bin(lambda x, y: x-y, foo1, foo2, foo1_kwargs, foo2_kwargs)


def Mult(foo1, foo2, foo1_kwargs=None, foo2_kwargs=None):
    return bin(lambda x, y: x*y, foo1, foo2, foo1_kwargs, foo2_kwargs)


def Div(foo1, foo2, foo1_kwargs=None, foo2_kwargs=None):
    return bin(lambda x, y: x/y, foo1, foo2, foo1_kwargs, foo2_kwargs)


def Mod(foo1, foo2, foo1_kwargs=None, foo2_kwargs=None):
    return bin(lambda x, y: x % y, foo1, foo2, foo1_kwargs, foo2_kwargs)


def Pow(foo1, foo2, foo1_kwargs=None, foo2_kwargs=None):
    return bin(lambda x, y: x**y, foo1, foo2, foo1_kwargs, foo2_kwargs)


def And(foo1, foo2, foo1_kwargs=None, foo2_kwargs=None):
    return bin(lambda x, y: x and y, foo1, foo2, foo1_kwargs, foo2_kwargs)


def Or(foo1, foo2, foo1_kwargs=None, foo2_kwargs=None):
    return bin(lambda x, y: x or y, foo1, foo2, foo1_kwargs, foo2_kwargs)
