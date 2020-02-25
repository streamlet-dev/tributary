import tributary.lazy as tl
import tributary.streaming as ts

from sympy.utilities.lambdify import lambdify
from sympy.parsing.sympy_parser import parse_expr, standard_transformations as _st, implicit_multiplication_application as _ima
from sympy import init_printing, dotprint, preorder_traversal
from graphviz import Source


init_printing(use_unicode=True)


def parse_expression(expr):
    '''Parse string as sympy expression
    Args:
        expr (string): string to convert to sympy expression
    '''
    return parse_expr(expr, transformations=(_st + (_ima,)))


def graphviz(expr):
    '''Plot sympy expression tree using graphviz
    Args:
        expr (sympy expression)
    '''

    return Source(dotprint(expr))


def traversal(expr):
    '''Traverse sympy expression tree
    Args:
        expr (sympy expression)
    '''

    return list(preorder_traversal(expr))


def symbols(expr):
    '''Get symbols used in sympy expression
    Args:
        expr (sympy expression)
    '''
    return expr.free_symbols


def construct_lazy(expr, modules=None):
    '''Construct Lazy tributary class from sympy expression

    Args:
        expr (sympy expression): A Sympy expression
        modules (list): a list of modules to use for sympy's lambdify function
    Returns:
        tributary.lazy.LazyGraph
    '''
    syms = list(symbols(expr))
    names = [s.name for s in syms]
    modules = modules or ["scipy", "numpy"]

    class Lazy(tl.LazyGraph):
        def __init__(self, **kwargs):
            for n in names:
                setattr(self, n, self.node(name=n, value=kwargs.get(n, None)))

            self._nodes = [getattr(self, n) for n in names]
            self._function = lambdify(syms, expr, modules=modules)
            self._expr = expr

        @tl.node
        def evaluate(self):
            return self._function(*self._nodes)

    return Lazy


def construct_streaming(expr, modules=None):
    '''Construct streaming tributary class from sympy expression

    Args:
        expr (sympy expression): A Sympy expression
        modules (list): a list of modules to use for sympy's lambdify function
    Returns:

    '''
    syms = list(symbols(expr))
    names = [s.name for s in syms]
    modules = modules or ["scipy", "numpy"]
    # modules = modules or  ["math", "mpmath", "sympy"]

    class Streaming(ts.StreamingGraph):
        def __init__(self, **kwargs):
            self._kwargs = {}
            for n in names:
                if n not in kwargs:
                    raise Exception("Must provide input source for: {}".format(n))
                setattr(self, n, kwargs.get(n))
                self._kwargs[n] = kwargs.get(n)

            self._nodes = [getattr(self, n) for n in names]
            self._lambda = lambdify(syms, expr, modules=modules)(**self._kwargs)
            self._expr = expr

            super(Streaming, self).__init__(output_node=self._lambda)

    return Streaming
