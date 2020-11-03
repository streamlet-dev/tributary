import math
import numpy as np
import scipy as sp
from .utils import _CALCULATIONS_GRAPHVIZSHAPE, _raise
from ..node import Node, _gen_node


def unary(foos, name):
    def _foo(self):
        foo = foos[0] if len(foos) == 1 or not self._use_dual else foos[1]
        downstream = Node(
            foo,
            {},
            name=name,
            inputs=1,
            use_dual=self._use_dual,
            graphvizshape=_CALCULATIONS_GRAPHVIZSHAPE,
        )
        self >> downstream
        return downstream

    return _foo


def binary(foos, name):
    def _foo(self, other):
        if not isinstance(other, Node):
            other = _gen_node(other)
            setattr(other, "_use_dual", self._use_dual)
        if self._use_dual != other._use_dual:
            raise NotImplementedError("Dual/Non-dual mismatch")
        foo = foos[0] if len(foos) == 1 or not self._use_dual else foos[1]
        downstream = Node(
            foo,
            {},
            name=name,
            inputs=2,
            use_dual=self._use_dual,
            graphvizshape=_CALCULATIONS_GRAPHVIZSHAPE,
        )
        self >> downstream
        other >> downstream
        return downstream

    return _foo


def n_ary(foos, name):
    def _foo(*others):
        use_dual = [x._use_dual for x in others]
        if np.any(use_dual) != np.all(use_dual):
            raise NotImplementedError("Dual/Non-dual mismatch")
        foo = foos[0] if len(foos) == 1 or not np.all(use_dual) else foos[1]
        downstream = Node(
            foo,
            {},
            name=name,
            inputs=len(others),
            graphvizshape=_CALCULATIONS_GRAPHVIZSHAPE,
        )
        for i, other in enumerate(others):
            other >> downstream
        return downstream

    return _foo


########################
# Arithmetic Operators #
########################
Noop = unary((lambda x: x,), name="Noop")
Negate = unary((lambda x: -1 * x, lambda x: (-1 * x[0], -1 * x[1])), name="Negate")
Invert = unary(
    (lambda x: 1 / x, lambda x: (1 / x[0], -x[1] / (x[0] ** 2))), name="Invert"
)
Add = binary((lambda x, y: x + y, lambda x, y: (x[0] + y[0], x[1] + y[1])), name="Add")
Sub = binary((lambda x, y: x - y, lambda x, y: (x[0] - y[0], x[1] - y[1])), name="Sub")
Mult = binary(
    (lambda x, y: x * y, lambda x, y: (x[0] * y[0], x[0] * y[1] + x[1] * y[0])),
    name="Mult",
)
Div = binary(
    (
        lambda x, y: x / y,
        lambda x, y: (x[0] / y[0], (x[1] * y[0] - x[0] * y[1]) / y[0] ** 2),
    ),
    name="Div",
)
RDiv = binary(
    (
        lambda x, y: y / x,
        lambda x, y: (y[0] / x[0], (y[1] * x[0] - y[0] * x[1]) / x[0] ** 2),
    ),
    name="RDiv",
)
Mod = binary(
    (lambda x, y: x % y, lambda: _raise(NotImplementedError("Not Implemented!"))),
    name="Mod",
)
Pow = binary(
    (lambda x, y: x ** y, lambda x, y: (x[0] ** y, y * x[1] * x[0] ** (y - 1))),
    name="Pow",
)
Sum = n_ary(
    (
        lambda *args: sum(args),
        lambda *args: (sum([x[0] for x in args]), sum(x[1] for x in args)),
    ),
    name="Sum",
)
Average = n_ary(
    (
        lambda *args: sum(args) / len(args),
        lambda *args: (
            (sum([x[0] for x in args]) / len(args), sum(x[1] for x in args) / len(args))
        ),
    ),
    name="Average",
)
Mean = Average

#####################
# Logical Operators #
#####################
Not = unary((lambda x: not x,), name="Not")
And = binary((lambda x, y: x and y,), name="And")
Or = binary((lambda x, y: x or y,), name="Or")


###################
# Numpy Functions #
###################
def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
    if ufunc == np.add:
        if isinstance(inputs[0], Node):
            return inputs[0].__add__(inputs[1])
        else:
            return inputs[1].__add__(inputs[0])
    elif ufunc == np.subtract:
        if isinstance(inputs[0], Node):
            return inputs[0].__sub__(inputs[1])
        else:
            return inputs[1].__sub__(inputs[0])
    elif ufunc == np.multiply:
        if isinstance(inputs[0], Node):
            return inputs[0].__mul__(inputs[1])
        else:
            return inputs[1].__mul__(inputs[0])
    elif ufunc == np.divide:
        if isinstance(inputs[0], Node):
            return inputs[0].__truediv__(inputs[1])
        else:
            return inputs[1].__truediv__(inputs[0])
    elif ufunc == np.sin:
        return inputs[0].sin()
    elif ufunc == np.cos:
        return inputs[0].cos()
    elif ufunc == np.tan:
        return inputs[0].tan()
    elif ufunc == np.arcsin:
        return inputs[0].asin()
    elif ufunc == np.arccos:
        return inputs[0].acos()
    elif ufunc == np.arctan:
        return inputs[0].atan()
    elif ufunc == np.exp:
        return inputs[0].exp()
    elif ufunc == sp.special.erf:
        return inputs[0].erf()
    raise NotImplementedError("Not Implemented!")


def __array_function__(self, func, method, *inputs, **kwargs):
    if func == np.lib.scimath.log:
        return inputs[0][0].log()
    elif func == np.lib.scimath.sqrt:
        return inputs[0][0].sqrt()
    raise NotImplementedError("Not Implemented!")


###############
# Comparators #
###############
Equal = binary((lambda x, y: x == y, lambda x, y: x[0] == y[0]), name="Equal")
NotEqual = binary((lambda x, y: x != y, lambda x, y: x[0] != y[0]), name="NotEqual")
Lt = binary((lambda x, y: x < y, lambda x, y: x[0] < y[0]), name="Less")
Le = binary(
    (lambda x, y: x <= y, lambda x, y: x[0] <= y[0] or x[0] == y[0]), name="LessOrEqual"
)
Gt = binary((lambda x, y: x > y, lambda x, y: x[0] > y[0]), name="Greater")
Ge = binary(
    (lambda x, y: x >= y, lambda x, y: x[0] > y[0] or x[0] == y[0]),
    name="GreaterOrEqual",
)


##########################
# Mathematical Functions #
##########################
Log = unary(
    (lambda x: math.log(x), lambda x: (math.log(x[0]), x[1] / x[0])), name="Log"
)
Sin = unary(
    (lambda x: math.sin(x), lambda x: (math.sin(x[0]), math.cos(x[0]) * x[1])),
    name="Sin",
)
Cos = unary(
    (lambda x: math.cos(x), lambda x: (math.cos(x[0]), -1 * math.sin(x[0]) * x[1])),
    name="Cos",
)
Tan = unary(
    (
        lambda x: math.tan(x),
        lambda x: (math.tan(x[0]), x[1] * (1 / math.cos(x[0])) ** 2),
    ),
    name="Tan",
)
Arcsin = unary(
    (
        lambda x: math.asin(x),
        lambda x: (math.asin(x[0]), x[1] / math.sqrt(1 - x[0] ** 2)),
    ),
    name="Arcsin",
)
Arccos = unary(
    (
        lambda x: math.acos(x),
        lambda x: (math.acos(x[0]), -1 * x[1] / math.sqrt(1 - x[0] ** 2)),
    ),
    name="Arccos",
)
Arctan = unary(
    (lambda x: math.atan(x), lambda x: (math.atan(x[0]), x[1] / (1 + x[0] ** 2))),
    name="Arctan",
)
Sqrt = unary(
    (lambda x: math.sqrt(x), lambda x: (math.sqrt(x[0]), x[1] * 0.5 / math.sqrt(x[0]))),
    name="Sqrt",
)
Abs = unary(
    (lambda x: abs(x), lambda x: (abs(x[0]), x[1] * x[0] / abs(x[0]))), name="Abs"
)
Exp = unary(
    (lambda x: math.exp(x), lambda x: (math.exp(x[0]), x[1] * math.exp(x[0]))),
    name="Exp",
)
Erf = unary(
    (
        lambda x: math.erf(x),
        lambda x: (
            math.erf(x[0]),
            x[1] * (2 / math.sqrt(math.pi)) * math.exp(-1 * math.pow(x[0], 2)),
        ),
    ),
    name="Erf",
)


##############
# Converters #
##############
Int = unary((lambda x: int(x), lambda x: int(x[0])), name="Int")
Float = unary((lambda x: float(x), lambda x: float(x[0])), name="Float")
Bool = unary((lambda x: bool(x), lambda x: bool(x[0])), name="Bool")
Str = unary((lambda x: str(x), lambda x: str(x[0]) + "+" + str(x[1]) + "ε"), name="Str")


# __Bool__ = unary(lambda x: bool(x), name='Noop')


def __Bool__(self):
    return True


###################
# Python Builtins #
###################
Floor = unary(
    (lambda x: math.floor(x), lambda x: (math.floor(x[0]), math.floor(x[1]))),
    name="Floor",
)
Ceil = unary(
    (lambda x: math.ceil(x), lambda x: (math.ceil(x[0]), math.ceil(x[1]))), name="Ceil"
)


def Round(self, ndigits=0):
    downstream = Node(
        lambda x: round(x, ndigits=ndigits)
        if not self._use_dual
        else (round(x[0], ndigits=ndigits), round(x[1], ndigits=ndigits)),
        {},
        name="Round",
        inputs=1,
        graphvizshape=_CALCULATIONS_GRAPHVIZSHAPE,
    )
    self.downstream().append((downstream, 0))
    downstream.upstream().append(self)
    return downstream


########################
# Arithmetic Operators #
########################
Node.__add__ = Add
Node.__radd__ = Add
Node.__sub__ = Sub
Node.__rsub__ = Sub
Node.__mul__ = Mult
Node.__rmul__ = Mult
# Node.__matmul__ = MatMult
# Node.__rmatmul__ = MatMult
Node.__div__ = Div
Node.__rdiv__ = RDiv
# Node.__divmod__ = DivMod
# Node.__rdivmod__ = DivMod
Node.__truediv__ = Div
Node.__rtruediv__ = RDiv
Node.__floordiv__ = Div
# Node.__lshift__ = LShift
# Node.__rlshift__ = LShift
# Node.__rshift__ = RShift
# Node.__rrshift__ = RShift

Node.__pow__ = Pow
Node.__rpow__ = Pow
Node.__mod__ = Mod
Node.__rmod__ = Mod

Node.sum = Sum
Node.average = Average
Node.mean = Average
Node.invert = Invert


#####################
# Logical Operators #
#####################
Node.__and__ = And
Node.__rand__ = And
Node.__or__ = Or
Node.__ror__ = Or
# Node.__xor__ = Xor
# Node.__rxor__ = Xor
Node.__invert__ = Not
Node.__bool__ = __Bool__

##############
# Converters #
##############
Node.int = Int
# Node.__int__ = Int
Node.float = Float
# Node.__float__ = Float
# Node.__str__ = Str
# Node.__complex__ = Int

###############
# Comparators #
###############
Node.__lt__ = Lt
Node.__le__ = Le
Node.__gt__ = Gt
Node.__ge__ = Ge
Node.__eq__ = Equal
Node.__ne__ = NotEqual
Node.__neg__ = Negate
# Node.__pos__ =
# Node.__nonzero__ = Bool  # Py2 compat

###################
# Python Builtins #
###################
# Node.__trunc__ = Len
Node.round = Round
Node.__round__ = Floor
Node.floor = Floor
Node.__floor__ = Floor
Node.ceil = Ceil
Node.__ceil__ = Ceil


###################
# Numpy Functions #
###################
Node.__array_ufunc__ = __array_ufunc__
Node.__array_function__ = __array_function__


##########################
# Mathematical Functions #
##########################
Node.log = Log
Node.sin = Sin
Node.cos = Cos
Node.tan = Tan
Node.asin = Arcsin
Node.acos = Arccos
Node.atan = Arctan
Node.abs = Abs
Node.sqrt = Sqrt
Node.exp = Exp
Node.erf = Erf
