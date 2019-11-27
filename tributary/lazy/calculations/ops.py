import math
import numpy as np
import scipy as sp
from ..base import _Node


def Add(self, other):
    other = self._tonode(other)
    if isinstance(self._self_reference, _Node):
        return self._gennode(self._name + '+' + other._name, (lambda x, y: x.value() + y.value()), [self._self_reference, other], self._trace or other._trace)
    return self._gennode(self._name + '+' + other._name, (lambda x, y: x.value() + y.value()), [self, other], self._trace or other._trace)


def Sub(self, other):
    other = self._tonode(other)
    if isinstance(self._self_reference, _Node):
        return self._gennode(self._name + '-' + other._name, (lambda x, y: x.value() - y.value()), [self._self_reference, other], self._trace or other._trace)
    return self._gennode(self._name + '-' + other._name, (lambda x, y: x.value() - y.value()), [self, other], self._trace or other._trace)


def Mult(self, other):
    other = self._tonode(other)
    if isinstance(self._self_reference, _Node):
        return self._gennode(self._name + '*' + other._name, (lambda x, y: x.value() * y.value()), [self._self_reference, other], self._trace or other._trace)
    return self._gennode(self._name + '*' + other._name, (lambda x, y: x.value() * y.value()), [self, other], self._trace or other._trace)


def Div(self, other):
    other = self._tonode(other)
    if isinstance(self._self_reference, _Node):
        return self._gennode(self._name + '/' + other._name, (lambda x, y: x.value() / y.value()), [self._self_reference, other], self._trace or other._trace)
    return self._gennode(self._name + '/' + other._name, (lambda x, y: x.value() / y.value()), [self, other], self._trace or other._trace)


def Pow(self, other):
    other = self._tonode(other)
    if isinstance(self._self_reference, _Node):
        return self._gennode(self._name + '^' + other._name, (lambda x, y: x.value() ** y.value()), [self._self_reference, other], self._trace or other._trace)
    return self._gennode(self._name + '^' + other._name, (lambda x, y: x.value() ** y.value()), [self, other], self._trace or other._trace)


def Sin(self):
    return self._gennode('sin(' + self._name + ')', (lambda x: math.sin(self.value())), [self], self._trace)


def Cos(self):
    return self._gennode('cos(' + self._name + ')', (lambda x: math.cos(self.value())), [self], self._trace)


def Tan(self):
    return self._gennode('tan(' + self._name + ')', (lambda x: math.tan(self.value())), [self], self._trace)


def Arcsin(self):
    return self._gennode('sin(' + self._name + ')', (lambda x: math.arcsin(self.value())), [self], self._trace)


def Arccos(self):
    return self._gennode('arccos(' + self._name + ')', (lambda x: math.arccos(self.value())), [self], self._trace)


def Arctan(self):
    return self._gennode('arctan(' + self._name + ')', (lambda x: math.arctan(self.value())), [self], self._trace)


def Sqrt(self):
    return self._gennode('sqrt(' + self._name + ')', (lambda x: math.sqrt(self.value())), [self], self._trace)


def Log(self):
    return self._gennode('log(' + self._name + ')', (lambda x: math.log(self.value())), [self], self._trace)


def Exp(self):
    return self._gennode('exp(' + self._name + ')', (lambda x: math.exp(self.value())), [self], self._trace)


def Erf(self):
    return self._gennode('erf(' + self._name + ')', (lambda x: math.erf(self.value())), [self], self._trace)


def Float(self):
    return self._gennode('float(' + self._name + ')', (lambda x: float(self.value())), [self], self._trace)


def Int(self):
    return self._gennode('int(' + self._name + ')', (lambda x: int(self.value())), [self], self._trace)


def Len(self):
    return self._gennode('len(' + self._name + ')', (lambda x: len(self.value())), [self], self._trace)


def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
    if ufunc == np.add:
        if isinstance(inputs[0], _Node):
            return inputs[0].__add__(inputs[1])
        else:
            return inputs[1].__add__(inputs[0])
    elif ufunc == np.subtract:
        if isinstance(inputs[0], _Node):
            return inputs[0].__sub__(inputs[1])
        else:
            return inputs[1].__sub__(inputs[0])
    elif ufunc == np.multiply:
        if isinstance(inputs[0], _Node):
            return inputs[0].__mul__(inputs[1])
        else:
            return inputs[1].__mul__(inputs[0])
    elif ufunc == np.divide:
        if isinstance(inputs[0], _Node):
            return inputs[0].__truedivide__(inputs[1])
        else:
            return inputs[1].__truedivide__(inputs[0])
    elif ufunc == np.sin:
        return inputs[0].sin()
    elif ufunc == np.cos:
        return inputs[0].cos()
    elif ufunc == np.tan:
        return inputs[0].tan()
    elif ufunc == np.arcsin:
        return inputs[0].arcsin()
    elif ufunc == np.arccos:
        return inputs[0].arccos()
    elif ufunc == np.arctan:
        return inputs[0].arctan()
    elif ufunc == np.exp:
        return inputs[0].exp()
    elif ufunc == sp.special.erf:
        return inputs[0].erf()
    else:
        raise NotImplementedError('Not Implemented!')


def Negate(self):
    return self._gennode('(-' + self._name + ')', (lambda x: -self.value()), [self], self._trace)


def Bool(self):
    if self.value() is None:
        return False
    return self.value()


def Equal(self, other):
    if isinstance(other, _Node) and super(_Node, self).__eq__(other):
        return True

    other = self._tonode(other)
    if isinstance(self._self_reference, _Node):
        return self._gennode(self._name + '==' + other._name, (lambda x, y: x() == y()), [self._self_reference, other], self._trace or other._trace)
    return self._gennode(self._name + '==' + other._name, (lambda x, y: x() == y()), [self, other], self._trace or other._trace)


def NotEqual(self, other):
    if isinstance(other, _Node) and super(_Node, self).__eq__(other):
        return False

    other = self._tonode(other)
    if isinstance(self._self_reference, _Node):
        return self._gennode(self._name + '!=' + other._name, (lambda x, y: x() != y()), [self._self_reference, other], self._trace or other._trace)
    return self._gennode(self._name + '!=' + other._name, (lambda x, y: x() != y()), [self, other], self._trace or other._trace)


def Ge(self, other):
    if isinstance(other, _Node) and super(_Node, self).__eq__(other):
        return True

    other = self._tonode(other)
    if isinstance(self._self_reference, _Node):
        return self._gennode(self._name + '>=' + other._name, (lambda x, y: x() >= y()), [self._self_reference, other], self._trace or other._trace)
    return self._gennode(self._name + '>=' + other._name, (lambda x, y: x() >= y()), [self, other], self._trace or other._trace)


def Gt(self, other):
    if isinstance(other, _Node) and super(_Node, self).__eq__(other):
        return False

    other = self._tonode(other)
    if isinstance(self._self_reference, _Node):
        return self._gennode(self._name + '>' + other._name, (lambda x, y: x() > y()), [self._self_reference, other], self._trace or other._trace)
    return self._gennode(self._name + '>' + other._name, (lambda x, y: x() > y()), [self, other], self._trace or other._trace)


def Le(self, other):
    if isinstance(other, _Node) and super(_Node, self).__eq__(other):
        return True

    other = self._tonode(other)
    if isinstance(self._self_reference, _Node):
        return self._gennode(self._name + '<=' + other._name, (lambda x, y: x() <= y()), [self._self_reference, other], self._trace or other._trace)
    return self._gennode(self._name + '<=' + other._name, (lambda x, y: x() <= y()), [self, other], self._trace or other._trace)


def Lt(self, other):
    if isinstance(other, _Node) and super(_Node, self).__eq__(other):
        return False
    other = self._tonode(other)
    if isinstance(self._self_reference, _Node):
        return self._gennode(self._name + '<' + other._name, (lambda x, y: x() < y()), [self._self_reference, other], self._trace or other._trace)
    return self._gennode(self._name + '<' + other._name, (lambda x, y: x() < y()), [self, other], self._trace or other._trace)


# Arithmetic
_Node.__add__ = Add
_Node.__radd__ = Add
_Node.__sub__ = Sub
_Node.__rsub__ = Sub
_Node.__mul__ = Mult
_Node.__rmul__ = Mult
_Node.__div__ = Div
_Node.__rdiv__ = Div
_Node.__truediv__ = Div
_Node.__rtruediv__ = Div

_Node.__pow__ = Pow
_Node.__rpow__ = Pow
# _Node.__mod__ = Mod
# _Node.__rmod__ = Mod

# Logical
# _Node.__and__ = And
# _Node.__or__ = Or
# _Node.__invert__ = Not
_Node.__bool__ = Bool

# Converters
_Node.int = Int
_Node.float = Float

# Comparator
_Node.__lt__ = Lt
_Node.__le__ = Le
_Node.__gt__ = Gt
_Node.__ge__ = Ge
_Node.__eq__ = Equal
_Node.__ne__ = NotEqual
_Node.__neg__ = Negate
_Node.__nonzero__ = Bool  # Py2 compat
_Node.__len__ = Len

# Numpy
_Node.__array_ufunc__ = __array_ufunc__

# Functions
_Node.log = Log
_Node.sin = Sin
_Node.cos = Cos
_Node.tan = Tan
_Node.arcsin = Arcsin
_Node.arccos = Arccos
_Node.arctan = Arctan
_Node.sqrt = Sqrt
_Node.exp = Exp
_Node.erf = Erf
