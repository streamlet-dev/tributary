import tributary.lazy as tl

from sympy.utilities.lambdify import lambdify
from sympy.parsing.sympy_parser import parse_expr, standard_transformations as _st, implicit_multiplication_application as _ima
from sympy import init_printing, dotprint, preorder_traversal
from graphviz import Source


init_printing(use_unicode=True)


def parse_expression(expr):
    return parse_expr(expr, transformations=(_st + (_ima,)))


def graphviz(expr):
    return Source(dotprint(expr))


def traversal(expr):
    return list(preorder_traversal(expr))


def symbols(expr):
    return expr.free_symbols


def construct_lazy(expr):
    syms = list(symbols(expr))
    names = [s.name for s in syms]

    class Lazy(tl.BaseClass):
        def __init__(self, **kwargs):
            print(names)
            for n in names:
                setattr(self, n, self.node(name=n, default_or_starting_value=kwargs.get(n, None)))

            self._nodes = [getattr(self, n) for n in names]
            self._function = lambdify(syms, expr)

        @tl.node
        def evaluate(self):
            return self._function(*self._nodes)

    return Lazy
