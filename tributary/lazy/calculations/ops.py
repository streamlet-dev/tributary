import math
import numpy as np
import scipy as sp
from .utils import _CALCULATIONS_GRAPHVIZSHAPE
from ..base import Node


def unary(node, name, lam):
    return node._gennode(name, lam, [node], node._trace, graphvizshape=_CALCULATIONS_GRAPHVIZSHAPE)


def binary(node1, other, name, lam):
    if isinstance(node1._self_reference, Node):
        return node1._gennode(name, lam, [node1._self_reference, other], node1._trace or other._trace, graphvizshape=_CALCULATIONS_GRAPHVIZSHAPE)
    return node1._gennode(name, lam, [node1, other], node1._trace or other._trace, graphvizshape=_CALCULATIONS_GRAPHVIZSHAPE)


def n_nary(node, others, name, lam):
    if isinstance(node._self_reference, Node):
        return node._gennode(name,
                             lam,
                             [node._self_reference] + others,
                             node._trace or any(other._trace for other in others),
                             graphvizshape=_CALCULATIONS_GRAPHVIZSHAPE)
    return node._gennode(name,
                         lam,
                         [node] + others,
                         node._trace or any(other._trace for other in others),
                         graphvizshape=_CALCULATIONS_GRAPHVIZSHAPE)

########################
# Arithmetic Operators #
########################


def Add(self, other):
    other = self._tonode(other)
    return binary(self, other, self._name_no_id() + '+' + other._name_no_id(), (lambda x, y: x.value() + y.value()))


def Sub(self, other):
    other = self._tonode(other)
    return binary(self, other, self._name_no_id() + '-' + other._name_no_id(), (lambda x, y: x.value() - y.value()))


def Mult(self, other):
    other = self._tonode(other)
    return binary(self, other, self._name_no_id() + '*' + other._name_no_id(), (lambda x, y: x.value() * y.value()))


def Div(self, other):
    other = self._tonode(other)
    return binary(self, other, self._name_no_id() + '/' + other._name_no_id(), (lambda x, y: x.value() / y.value()))


def Pow(self, other):
    other = self._tonode(other)
    return binary(self, other, self._name_no_id() + '^' + other._name_no_id(), (lambda x, y: x.value() ** y.value()))


def Mod(self, other):
    other = self._tonode(other)
    return binary(self, other, self._name_no_id() + '%' + other._name_no_id(), (lambda x, y: x.value() % y.value()))


def Negate(self):
    return unary(self, '(-' + self._name_no_id() + ')', (lambda x: -self.value()))


def Invert(self):
    return unary(self, '1/' + self._name_no_id(), (lambda x: 1 / self.value()))


def Sum(self, *others):
    others_nodes = []
    for other in others:
        others_nodes.append(self._tonode(other))

    return n_nary(self,
                  others_nodes,
                  'Sum(' + self._name_no_id() + ',' + ','.join(other._name_no_id() for other in others_nodes) + ')',
                  (lambda *args: sum(x.value() for x in args)))


def Average(self, *others):
    others_nodes = []
    for other in others:
        others_nodes.append(self._tonode(other))

    return n_nary(self,
                  others_nodes,
                  'Average(' + self._name_no_id() + ',' + ','.join(other._name_no_id() for other in others_nodes) + ')',
                  (lambda *args: sum(x.value() for x in args) / len(args)))


#####################
# Logical Operators #
#####################
def Or(self, other):
    other = self._tonode(other)
    return binary(self, other, self._name_no_id() + '||' + other._name_no_id(), (lambda x, y: x.value() or y.value()))


def And(self, other):
    other = self._tonode(other)
    return binary(self, other, self._name_no_id() + '&&' + other._name_no_id(), (lambda x, y: x.value() and y.value()))


def Not(self):
    return unary(self, '!' + self._name_no_id(), (lambda x: not x.value()))


##########################
# Mathematical Functions #
##########################
def Sin(self):
    return unary(self, 'sin(' + self._name_no_id() + ')', (lambda x: math.sin(self.value())))


def Cos(self):
    return unary(self, 'cos(' + self._name_no_id() + ')', (lambda x: math.cos(self.value())))


def Tan(self):
    return unary(self, 'tan(' + self._name_no_id() + ')', (lambda x: math.tan(self.value())))


def Arcsin(self):
    return unary(self, 'arcsin(' + self._name_no_id() + ')', (lambda x: math.asin(self.value())))


def Arccos(self):
    return unary(self, 'arccos(' + self._name_no_id() + ')', (lambda x: math.acos(self.value())))


def Arctan(self):
    return unary(self, 'arctan(' + self._name_no_id() + ')', (lambda x: math.atan(self.value())))


def Abs(self):
    return unary(self, '||' + self._name_no_id() + '||', (lambda x: abs(self.value())))


def Sqrt(self):
    return unary(self, 'sqrt(' + str(self._name_no_id()) + ')', (lambda x: math.sqrt(self.value())))


def Log(self):
    return unary(self, 'log(' + str(self._name_no_id()) + ')', (lambda x: math.log(self.value())))


def Exp(self):
    return unary(self, 'exp(' + str(self._name_no_id()) + ')', (lambda x: math.exp(self.value())))


def Erf(self):
    return unary(self, 'erf(' + str(self._name_no_id()) + ')', (lambda x: math.erf(self.value())))


##############
# Converters #
##############
def Float(self):
    return unary(self, 'float(' + str(self._name_no_id()) + ')', (lambda x: float(self.value())))


def Int(self):
    return unary(self, 'int(' + str(self._name_no_id()) + ')', (lambda x: int(self.value())))


def Bool(self):
    return unary(self, 'bool(' + str(self._name_no_id()) + ')', (lambda x: bool(self.value())))


def __Bool__(self):
    if self.value() is None:
        return False
    return bool(self.value())


def Str(self):
    return unary(self, 'str(' + str(self._name_no_id()) + ')', (lambda x: str(self.value())))


###################
# Python Builtins #
###################
def Len(self):
    return unary(self, 'len(' + str(self._name_no_id()) + ')', (lambda x: len(self.value())))


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
def Equal(self, other):
    if isinstance(other, Node) and super(Node, self).__eq__(other):
        return True
    other = self._tonode(other)
    return binary(self, other, self._name_no_id() + '==' + other._name_no_id(), (lambda x, y: x.value() == y.value()))


def NotEqual(self, other):
    if isinstance(other, Node) and super(Node, self).__eq__(other):
        return False
    other = self._tonode(other)
    return binary(self, other, self._name_no_id() + '!=' + other._name_no_id(), (lambda x, y: x.value() != y.value()))


def Ge(self, other):
    if isinstance(other, Node) and super(Node, self).__eq__(other):
        return True
    other = self._tonode(other)
    return binary(self, other, self._name_no_id() + '>=' + other._name_no_id(), (lambda x, y: x.value() >= y.value()))


def Gt(self, other):
    if isinstance(other, Node) and super(Node, self).__eq__(other):
        return False
    other = self._tonode(other)
    return binary(self, other, self._name_no_id() + '>' + other._name_no_id(), (lambda x, y: x.value() > y.value()))


def Le(self, other):
    if isinstance(other, Node) and super(Node, self).__eq__(other):
        return True
    other = self._tonode(other)
    return binary(self, other, self._name_no_id() + '<=' + other._name_no_id(), (lambda x, y: x.value() <= y.value()))


def Lt(self, other):
    if isinstance(other, Node) and super(Node, self).__eq__(other):
        return False
    other = self._tonode(other)
    return binary(self, other, self._name_no_id() + '<' + other._name_no_id(), (lambda x, y: x.value() < y.value()))


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
Node.__rdiv__ = Div
# Node.__divmod__ = DivMod
# Node.__rdivmod__ = DivMod
Node.__truediv__ = Div
Node.__rtruediv__ = Div
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

##############
# Converters #
##############
Node.int = Int
Node.float = Float
Node.__bool__ = __Bool__

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
# Node.__pos__ = Pos
Node.__nonzero__ = Bool  # Py2 compat

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
