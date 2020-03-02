import math
import numpy as np
import scipy as sp
from ..base import Node


########################
# Arithmetic Operators #
########################
def Add(self, other):
    other = self._tonode(other)
    if isinstance(self._self_reference, Node):
        return self._gennode(self._name_no_id() + '+' + other._name_no_id(), (lambda x, y: x.value() + y.value()), [self._self_reference, other], self._trace or other._trace)
    return self._gennode(self._name_no_id() + '+' + other._name_no_id(), (lambda x, y: x.value() + y.value()), [self, other], self._trace or other._trace)


def Sub(self, other):
    other = self._tonode(other)
    if isinstance(self._self_reference, Node):
        return self._gennode(self._name_no_id() + '-' + other._name_no_id(), (lambda x, y: x.value() - y.value()), [self._self_reference, other], self._trace or other._trace)
    return self._gennode(self._name_no_id() + '-' + other._name_no_id(), (lambda x, y: x.value() - y.value()), [self, other], self._trace or other._trace)


def Mult(self, other):
    other = self._tonode(other)
    if isinstance(self._self_reference, Node):
        return self._gennode(self._name_no_id() + '*' + other._name_no_id(), (lambda x, y: x.value() * y.value()), [self._self_reference, other], self._trace or other._trace)
    return self._gennode(self._name_no_id() + '*' + other._name_no_id(), (lambda x, y: x.value() * y.value()), [self, other], self._trace or other._trace)


def Div(self, other):
    other = self._tonode(other)
    if isinstance(self._self_reference, Node):
        return self._gennode(self._name_no_id() + '/' + other._name_no_id(), (lambda x, y: x.value() / y.value()), [self._self_reference, other], self._trace or other._trace)
    return self._gennode(self._name_no_id() + '/' + other._name_no_id(), (lambda x, y: x.value() / y.value()), [self, other], self._trace or other._trace)


def Pow(self, other):
    other = self._tonode(other)
    if isinstance(self._self_reference, Node):
        return self._gennode(self._name_no_id() + '^' + other._name_no_id(), (lambda x, y: x.value() ** y.value()), [self._self_reference, other], self._trace or other._trace)
    return self._gennode(self._name_no_id() + '^' + other._name_no_id(), (lambda x, y: x.value() ** y.value()), [self, other], self._trace or other._trace)


def Mod(self, other):
    other = self._tonode(other)
    if isinstance(self._self_reference, Node):
        return self._gennode(self._name_no_id() + '%' + other._name_no_id(), (lambda x, y: x.value() % y.value()), [self._self_reference, other], self._trace or other._trace)
    return self._gennode(self._name_no_id() + '%' + other._name_no_id(), (lambda x, y: x.value() % y.value()), [self, other], self._trace or other._trace)


def Negate(self):
    return self._gennode('(-' + self._name_no_id() + ')', (lambda x: -self.value()), [self], self._trace)


def Invert(self):
    return self._gennode('1/' + self._name_no_id(), (lambda x: 1 / self.value()), [self], self._trace)


def Sum(self, *others):
    others_nodes = []
    for other in others:
        others_nodes.append(self._tonode(other))
    if isinstance(self._self_reference, Node):
        return self._gennode('Sum(' + self._name_no_id() + ',' + ','.join(other._name_no_id() for other in others_nodes) + ')',
                             lambda *args: sum(x.value() for x in args),
                             [self._self_reference] + others_nodes,
                             self._trace or any(other._trace for other in others_nodes))
    return self._gennode('Sum(' + self._name_no_id() + ',' + ','.join(other._name_no_id() for other in others_nodes) + ')',
                         lambda *args: sum(x.value() for x in args),
                         [self] + others_nodes,
                         self._trace or any(other._trace for other in others_nodes))


def Average(self, *others):
    others_nodes = []
    for other in others:
        others_nodes.append(self._tonode(other))
    if isinstance(self._self_reference, Node):
        return self._gennode('Average(' + self._name_no_id() + ',' + ','.join(other._name_no_id() for other in others_nodes) + ')',
                             lambda *args: sum(x.value() for x in args) / len(args),
                             [self._self_reference] + others_nodes,
                             self._trace or any(other._trace for other in others_nodes))
    return self._gennode('Average(' + self._name_no_id() + ',' + ','.join(other._name_no_id() for other in others_nodes) + ')',
                         lambda *args: sum(x.value() for x in args) / len(args),
                         [self] + others_nodes,
                         self._trace or any(other._trace for other in others_nodes))


#####################
# Logical Operators #
#####################
def Or(self, other):
    other = self._tonode(other)
    if isinstance(self._self_reference, Node):
        return self._gennode(self._name_no_id() + '||' + other._name_no_id(), (lambda x, y: x.value() or y.value()), [self._self_reference, other], self._trace or other._trace)
    return self._gennode(self._name_no_id() + '||' + other._name_no_id(), (lambda x, y: x.value() or y.value()), [self, other], self._trace or other._trace)


def And(self, other):
    other = self._tonode(other)
    if isinstance(self._self_reference, Node):
        return self._gennode(self._name_no_id() + '&&' + other._name_no_id(), (lambda x, y: x.value() or y.value()), [self._self_reference, other], self._trace or other._trace)
    return self._gennode(self._name_no_id() + '&&' + other._name_no_id(), (lambda x, y: x.value() and y.value()), [self, other], self._trace or other._trace)


def Not(self):
    return self._gennode('!' + self._name_no_id(), (lambda x: not x.value()), [self], self._trace)


##########################
# Mathematical Functions #
##########################
def Sin(self):
    return self._gennode('sin(' + self._name_no_id() + ')', (lambda x: math.sin(self.value())), [self], self._trace)


def Cos(self):
    return self._gennode('cos(' + self._name_no_id() + ')', (lambda x: math.cos(self.value())), [self], self._trace)


def Tan(self):
    return self._gennode('tan(' + self._name_no_id() + ')', (lambda x: math.tan(self.value())), [self], self._trace)


def Arcsin(self):
    return self._gennode('arcsin(' + self._name_no_id() + ')', (lambda x: math.asin(self.value())), [self], self._trace)


def Arccos(self):
    return self._gennode('arccos(' + self._name_no_id() + ')', (lambda x: math.acos(self.value())), [self], self._trace)


def Arctan(self):
    return self._gennode('arctan(' + self._name_no_id() + ')', (lambda x: math.atan(self.value())), [self], self._trace)


def Abs(self):
    return self._gennode('||' + self._name_no_id() + '||', (lambda x: abs(self.value())), [self], self._trace)


def Sqrt(self):
    return self._gennode('sqrt(' + str(self._name_no_id()) + ')', (lambda x: math.sqrt(self.value())), [self], self._trace)


def Log(self):
    return self._gennode('log(' + str(self._name_no_id()) + ')', (lambda x: math.log(self.value())), [self], self._trace)


def Exp(self):
    return self._gennode('exp(' + str(self._name_no_id()) + ')', (lambda x: math.exp(self.value())), [self], self._trace)


def Erf(self):
    return self._gennode('erf(' + str(self._name_no_id()) + ')', (lambda x: math.erf(self.value())), [self], self._trace)


##############
# Converters #
##############
def Float(self):
    return self._gennode('float(' + str(self._name_no_id()) + ')', (lambda x: float(self.value())), [self], self._trace)


def Int(self):
    return self._gennode('int(' + str(self._name_no_id()) + ')', (lambda x: int(self.value())), [self], self._trace)


def Bool(self):
    return self._gennode('bool(' + str(self._name_no_id()) + ')', (lambda x: bool(self.value())), [self], self._trace)


def __Bool__(self):
    if self.value() is None:
        return False
    return bool(self.value())


def Str(self):
    return self._gennode('str(' + str(self._name_no_id()) + ')', (lambda x: str(self.value())), [self], self._trace)


###################
# Python Builtins #
###################
def Len(self):
    return self._gennode('len(' + str(self._name_no_id()) + ')', (lambda x: len(self.value())), [self], self._trace)


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
    if isinstance(self._self_reference, Node):
        return self._gennode(str(self._name_no_id()) + '==' + other._name_no_id(), (lambda x, y: x() == y()), [self._self_reference, other], self._trace or other._trace)
    return self._gennode(str(self._name_no_id()) + '==' + other._name_no_id(), (lambda x, y: x() == y()), [self, other], self._trace or other._trace)


def NotEqual(self, other):
    if isinstance(other, Node) and super(Node, self).__eq__(other):
        return False

    other = self._tonode(other)
    if isinstance(self._self_reference, Node):
        return self._gennode(str(self._name_no_id()) + '!=' + other._name_no_id(), (lambda x, y: x() != y()), [self._self_reference, other], self._trace or other._trace)
    return self._gennode(str(self._name_no_id()) + '!=' + other._name_no_id(), (lambda x, y: x() != y()), [self, other], self._trace or other._trace)


def Ge(self, other):
    if isinstance(other, Node) and super(Node, self).__eq__(other):
        return True

    other = self._tonode(other)
    if isinstance(self._self_reference, Node):
        return self._gennode(str(self._name_no_id()) + '>=' + other._name_no_id(), (lambda x, y: x() >= y()), [self._self_reference, other], self._trace or other._trace)
    return self._gennode(str(self._name_no_id()) + '>=' + other._name_no_id(), (lambda x, y: x() >= y()), [self, other], self._trace or other._trace)


def Gt(self, other):
    if isinstance(other, Node) and super(Node, self).__eq__(other):
        return False

    other = self._tonode(other)
    if isinstance(self._self_reference, Node):
        return self._gennode(str(self._name_no_id()) + '>' + other._name_no_id(), (lambda x, y: x() > y()), [self._self_reference, other], self._trace or other._trace)
    return self._gennode(str(self._name_no_id()) + '>' + other._name_no_id(), (lambda x, y: x() > y()), [self, other], self._trace or other._trace)


def Le(self, other):
    if isinstance(other, Node) and super(Node, self).__eq__(other):
        return True

    other = self._tonode(other)
    if isinstance(self._self_reference, Node):
        return self._gennode(str(self._name_no_id()) + '<=' + other._name_no_id(), (lambda x, y: x() <= y()), [self._self_reference, other], self._trace or other._trace)
    return self._gennode(str(self._name_no_id()) + '<=' + other._name_no_id(), (lambda x, y: x() <= y()), [self, other], self._trace or other._trace)


def Lt(self, other):
    if isinstance(other, Node) and super(Node, self).__eq__(other):
        return False
    other = self._tonode(other)
    if isinstance(self._self_reference, Node):
        return self._gennode(str(self._name_no_id()) + '<' + other._name_no_id(), (lambda x, y: x() < y()), [self._self_reference, other], self._trace or other._trace)
    return self._gennode(str(self._name_no_id()) + '<' + other._name_no_id(), (lambda x, y: x() < y()), [self, other], self._trace or other._trace)


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
