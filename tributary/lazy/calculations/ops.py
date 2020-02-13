import math
import numpy as np
import scipy as sp
from ..base import BaseNode


########################
# Arithmetic Operators #
########################
def Add(self, other):
    other = self._tonode(other)
    if isinstance(self._self_reference, BaseNode):
        return self._gennode(self._name + '+' + other._name, (lambda x, y: x.value() + y.value()), [self._self_reference, other], self._trace or other._trace)
    return self._gennode(self._name + '+' + other._name, (lambda x, y: x.value() + y.value()), [self, other], self._trace or other._trace)


def Sub(self, other):
    other = self._tonode(other)
    if isinstance(self._self_reference, BaseNode):
        return self._gennode(self._name + '-' + other._name, (lambda x, y: x.value() - y.value()), [self._self_reference, other], self._trace or other._trace)
    return self._gennode(self._name + '-' + other._name, (lambda x, y: x.value() - y.value()), [self, other], self._trace or other._trace)


def Mult(self, other):
    other = self._tonode(other)
    if isinstance(self._self_reference, BaseNode):
        return self._gennode(self._name + '*' + other._name, (lambda x, y: x.value() * y.value()), [self._self_reference, other], self._trace or other._trace)
    return self._gennode(self._name + '*' + other._name, (lambda x, y: x.value() * y.value()), [self, other], self._trace or other._trace)


def Div(self, other):
    other = self._tonode(other)
    if isinstance(self._self_reference, BaseNode):
        return self._gennode(self._name + '/' + other._name, (lambda x, y: x.value() / y.value()), [self._self_reference, other], self._trace or other._trace)
    return self._gennode(self._name + '/' + other._name, (lambda x, y: x.value() / y.value()), [self, other], self._trace or other._trace)


def Pow(self, other):
    other = self._tonode(other)
    if isinstance(self._self_reference, BaseNode):
        return self._gennode(self._name + '^' + other._name, (lambda x, y: x.value() ** y.value()), [self._self_reference, other], self._trace or other._trace)
    return self._gennode(self._name + '^' + other._name, (lambda x, y: x.value() ** y.value()), [self, other], self._trace or other._trace)


def Negate(self):
    return self._gennode('(-' + self._name + ')', (lambda x: -self.value()), [self], self._trace)


#####################
# Logical Operators #
#####################
def Or(self, other):
    other = self._tonode(other)
    if isinstance(self._self_reference, BaseNode):
        return self._gennode(self._name + '||' + other._name, (lambda x, y: x.value() or y.value()), [self._self_reference, other], self._trace or other._trace)
    return self._gennode(self._name + '||' + other._name, (lambda x, y: x.value() or y.value()), [self, other], self._trace or other._trace)


def And(self, other):
    other = self._tonode(other)
    if isinstance(self._self_reference, BaseNode):
        return self._gennode(self._name + '&&' + other._name, (lambda x, y: x.value() or y.value()), [self._self_reference, other], self._trace or other._trace)
    return self._gennode(self._name + '&&' + other._name, (lambda x, y: x.value() and y.value()), [self, other], self._trace or other._trace)


def Not(self):
    return self._gennode('!' + self._name, (lambda x: not x.value()), [self], self._trace)


##########################
# Mathematical Functions #
##########################
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


##############
# Converters #
##############
def Float(self):
    return self._gennode('float(' + self._name + ')', (lambda x: float(self.value())), [self], self._trace)


def Int(self):
    return self._gennode('int(' + self._name + ')', (lambda x: int(self.value())), [self], self._trace)


def Bool(self):
    if self.value() is None:
        return False
    return bool(self.value())


###################
# Python Builtins #
###################
def Len(self):
    return self._gennode('len(' + self._name + ')', (lambda x: len(self.value())), [self], self._trace)


###################
# Numpy Functions #
###################
def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
    if ufunc == np.add:
        if isinstance(inputs[0], BaseNode):
            return inputs[0].__add__(inputs[1])
        else:
            return inputs[1].__add__(inputs[0])
    elif ufunc == np.subtract:
        if isinstance(inputs[0], BaseNode):
            return inputs[0].__sub__(inputs[1])
        else:
            return inputs[1].__sub__(inputs[0])
    elif ufunc == np.multiply:
        if isinstance(inputs[0], BaseNode):
            return inputs[0].__mul__(inputs[1])
        else:
            return inputs[1].__mul__(inputs[0])
    elif ufunc == np.divide:
        if isinstance(inputs[0], BaseNode):
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


###############
# Comparators #
###############
def Equal(self, other):
    if isinstance(other, BaseNode) and super(BaseNode, self).__eq__(other):
        return True

    other = self._tonode(other)
    if isinstance(self._self_reference, BaseNode):
        return self._gennode(self._name + '==' + other._name, (lambda x, y: x() == y()), [self._self_reference, other], self._trace or other._trace)
    return self._gennode(self._name + '==' + other._name, (lambda x, y: x() == y()), [self, other], self._trace or other._trace)


def NotEqual(self, other):
    if isinstance(other, BaseNode) and super(BaseNode, self).__eq__(other):
        return False

    other = self._tonode(other)
    if isinstance(self._self_reference, BaseNode):
        return self._gennode(self._name + '!=' + other._name, (lambda x, y: x() != y()), [self._self_reference, other], self._trace or other._trace)
    return self._gennode(self._name + '!=' + other._name, (lambda x, y: x() != y()), [self, other], self._trace or other._trace)


def Ge(self, other):
    if isinstance(other, BaseNode) and super(BaseNode, self).__eq__(other):
        return True

    other = self._tonode(other)
    if isinstance(self._self_reference, BaseNode):
        return self._gennode(self._name + '>=' + other._name, (lambda x, y: x() >= y()), [self._self_reference, other], self._trace or other._trace)
    return self._gennode(self._name + '>=' + other._name, (lambda x, y: x() >= y()), [self, other], self._trace or other._trace)


def Gt(self, other):
    if isinstance(other, BaseNode) and super(BaseNode, self).__eq__(other):
        return False

    other = self._tonode(other)
    if isinstance(self._self_reference, BaseNode):
        return self._gennode(self._name + '>' + other._name, (lambda x, y: x() > y()), [self._self_reference, other], self._trace or other._trace)
    return self._gennode(self._name + '>' + other._name, (lambda x, y: x() > y()), [self, other], self._trace or other._trace)


def Le(self, other):
    if isinstance(other, BaseNode) and super(BaseNode, self).__eq__(other):
        return True

    other = self._tonode(other)
    if isinstance(self._self_reference, BaseNode):
        return self._gennode(self._name + '<=' + other._name, (lambda x, y: x() <= y()), [self._self_reference, other], self._trace or other._trace)
    return self._gennode(self._name + '<=' + other._name, (lambda x, y: x() <= y()), [self, other], self._trace or other._trace)


def Lt(self, other):
    if isinstance(other, BaseNode) and super(BaseNode, self).__eq__(other):
        return False
    other = self._tonode(other)
    if isinstance(self._self_reference, BaseNode):
        return self._gennode(self._name + '<' + other._name, (lambda x, y: x() < y()), [self._self_reference, other], self._trace or other._trace)
    return self._gennode(self._name + '<' + other._name, (lambda x, y: x() < y()), [self, other], self._trace or other._trace)


########################
# Arithmetic Operators #
########################
BaseNode.__add__ = Add
BaseNode.__radd__ = Add
BaseNode.__sub__ = Sub
BaseNode.__rsub__ = Sub
BaseNode.__mul__ = Mult
BaseNode.__rmul__ = Mult
BaseNode.__div__ = Div
BaseNode.__rdiv__ = Div
BaseNode.__truediv__ = Div
BaseNode.__rtruediv__ = Div

BaseNode.__pow__ = Pow
BaseNode.__rpow__ = Pow
# BaseNode.__mod__ = Mod
# BaseNode.__rmod__ = Mod

#####################
# Logical Operators #
#####################
BaseNode.__and__ = And
BaseNode.__or__ = Or
BaseNode.__invert__ = Not

##############
# Converters #
##############
BaseNode.int = Int
BaseNode.float = Float
BaseNode.__bool__ = Bool

###############
# Comparators #
###############
BaseNode.__lt__ = Lt
BaseNode.__le__ = Le
BaseNode.__gt__ = Gt
BaseNode.__ge__ = Ge
BaseNode.__eq__ = Equal
BaseNode.__ne__ = NotEqual
BaseNode.__neg__ = Negate
BaseNode.__nonzero__ = Bool  # Py2 compat

###################
# Python Builtins #
###################
BaseNode.__len__ = Len

###################
# Numpy Functions #
###################
BaseNode.__array_ufunc__ = __array_ufunc__

##########################
# Mathematical Functions #
##########################
BaseNode.log = Log
BaseNode.sin = Sin
BaseNode.cos = Cos
BaseNode.tan = Tan
BaseNode.arcsin = Arcsin
BaseNode.arccos = Arccos
BaseNode.arctan = Arctan
BaseNode.sqrt = Sqrt
BaseNode.exp = Exp
BaseNode.erf = Erf
