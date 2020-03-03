import math
import numpy as np
import scipy as sp
from .utils import _CALCULATIONS_GRAPHVIZSHAPE
from ..base import Node, _gen_node


def unary(foo, name):
    def _foo(self):
        downstream = Node(foo, {}, name=name, inputs=1, graphvizshape=_CALCULATIONS_GRAPHVIZSHAPE)
        self._downstream.append((downstream, 0))
        downstream._upstream.append(self)
        return downstream
    return _foo


def binary(foo, name):
    def _foo(self, other):
        other = _gen_node(other)
        downstream = Node(foo, {}, name=name, inputs=2, graphvizshape=_CALCULATIONS_GRAPHVIZSHAPE)
        self._downstream.append((downstream, 0))
        other._downstream.append((downstream, 1))
        downstream._upstream.extend([self, other])
        return downstream
    return _foo


def n_ary(foo, name):
    def _foo(*others):
        downstream = Node(foo, {}, name=name, inputs=len(others), graphvizshape=_CALCULATIONS_GRAPHVIZSHAPE)
        for i, other in enumerate(others):
            other._downstream.append((downstream, i))

        downstream._upstream.extend(list(others))
        return downstream
    return _foo


########################
# Arithmetic Operators #
########################
Noop = unary(lambda x: x, name='Noop')
Negate = unary(lambda x: -1 * x, name='Negate')
Invert = unary(lambda x: 1 / x, name='Invert')
Add = binary(lambda x, y: x + y, name='Add')
Sub = binary(lambda x, y: x - y, name='Sub')
Mult = binary(lambda x, y: x * y, name='Mult')
Div = binary(lambda x, y: x / y, name='Div')
RDiv = binary(lambda x, y: y / x, name='RDiv')
Mod = binary(lambda x, y: x % y, name='Mod')
Pow = binary(lambda x, y: x ** y, name='Pow')
Sum = n_ary(lambda *args: sum(args), name='Sum')
Average = n_ary(lambda *args: sum(args) / len(args), name='Average')

#####################
# Logical Operators #
#####################
Not = unary(lambda x: not x, name='Not')
And = binary(lambda x, y: x and y, name='And')
Or = binary(lambda x, y: x or y, name='Or')


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
    raise NotImplementedError('Not Implemented!')


def __array_function__(self, func, method, *inputs, **kwargs):
    if func == np.lib.scimath.log:
        return inputs[0][0].log()
    elif func == np.lib.scimath.sqrt:
        return inputs[0][0].sqrt()
    raise NotImplementedError('Not Implemented!')


###############
# Comparators #
###############
Equal = binary(lambda x, y: x == y, name='Equal')
NotEqual = binary(lambda x, y: x != y, name='NotEqual')
Lt = binary(lambda x, y: x < y, name='Less')
Le = binary(lambda x, y: x <= y, name='LessOrEqual')
Gt = binary(lambda x, y: x > y, name='Greater')
Ge = binary(lambda x, y: x >= y, name='GreaterOrEqual')


##########################
# Mathematical Functions #
##########################
Log = unary(lambda x: math.log(x), name='Log')
Sin = unary(lambda x: math.sin(x), name='Sin')
Cos = unary(lambda x: math.cos(x), name='Cos')
Tan = unary(lambda x: math.tan(x), name='Tan')
Arcsin = unary(lambda x: math.asin(x), name='Arcsin')
Arccos = unary(lambda x: math.acos(x), name='Arccos')
Arctan = unary(lambda x: math.atan(x), name='Arctan')
Sqrt = unary(lambda x: math.sqrt(x), name='Sqrt')
Abs = unary(lambda x: abs(x), name='Abs')
Exp = unary(lambda x: math.exp(x), name='Exp')
Erf = unary(lambda x: math.erf(x), name='Erf')


##############
# Converters #
##############
Int = unary(lambda x: int(x), name='Int')
Float = unary(lambda x: float(x), name='Float')
Bool = unary(lambda x: bool(x), name='Bool')
Str = unary(lambda x: str(x), name='Str')


# __Bool__ = unary(lambda x: bool(x), name='Noop')

def __Bool__(self):
    return True


###################
# Python Builtins #
###################
Len = unary(lambda x: len(x), name='Len')


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
Node.__len__ = Len
# Node.__round__ = Len
# Node.__trunc__ = Len
# Node.__floor__ = Len
# Node.__ceil__ = Len

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
