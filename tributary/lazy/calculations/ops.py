import math
import numpy as np
import scipy as sp
from .utils import _CALCULATIONS_GRAPHVIZSHAPE
from ..node import Node


def unary(node, name, lam):
    return node._gennode(
        name,
        lam,
        [node],
        graphvizshape=_CALCULATIONS_GRAPHVIZSHAPE,
        use_dual=node._use_dual,
    )


def binary(node1, other, name, lam):
    if isinstance(node1._self_reference, Node):
        return node1._gennode(
            name,
            lam,
            [node1._self_reference, other],
            graphvizshape=_CALCULATIONS_GRAPHVIZSHAPE,
            use_dual=node1._use_dual,
        )
    return node1._gennode(
        name,
        lam,
        [node1, other],
        graphvizshape=_CALCULATIONS_GRAPHVIZSHAPE,
        use_dual=node1._use_dual,
    )


def n_nary(node, others, name, lam):
    if isinstance(node._self_reference, Node):
        return node._gennode(
            name,
            lam,
            [node._self_reference] + others,
            graphvizshape=_CALCULATIONS_GRAPHVIZSHAPE,
            use_dual=node._use_dual,
        )
    return node._gennode(
        name,
        lam,
        [node] + others,
        graphvizshape=_CALCULATIONS_GRAPHVIZSHAPE,
        use_dual=node._use_dual,
    )


########################
# Arithmetic Operators #
########################


def Add(self, other):
    """Compute n1 + n2 for nodes n1, n2"""
    other = self._tonode(other)
    other._use_dual = self._use_dual
    return binary(
        self,
        other,
        "{}+{}".format(self._name_no_id(), other._name_no_id()),
        (
            lambda x, y: x.value() + y.value()
            if not self._use_dual
            else (x.value()[0] + y.value()[0], x.value()[1] + y.value()[1])
        ),
    )


def Sub(self, other):
    """Compute n1 - n2 for nodes n1, n2"""
    other = self._tonode(other)
    other._use_dual = self._use_dual
    return binary(
        self,
        other,
        "{}-{}".format(self._name_no_id(), other._name_no_id()),
        (
            lambda x, y: x.value() - y.value()
            if not self._use_dual
            else (x.value()[0] - y.value()[0], x.value()[1] - y.value()[1])
        ),
    )


def Mult(self, other):
    """Compute n1 * n2 for nodes n1, n2"""
    other = self._tonode(other)
    other._use_dual = self._use_dual
    return binary(
        self,
        other,
        "{}*{}".format(self._name_no_id(), other._name_no_id()),
        (
            lambda x, y: x.value() * y.value()
            if not self._use_dual
            else (
                x.value()[0] * y.value()[0],
                x.value()[0] * y.value()[1] + x.value()[1] * y.value()[0],
            )
        ),
    )


def Div(self, other):
    """Compute n1 / n2 for nodes n1, n2"""
    other = self._tonode(other)
    other._use_dual = self._use_dual
    return binary(
        self,
        other,
        "{}/{}".format(self._name_no_id(), other._name_no_id()),
        (
            lambda x, y: x.value() / y.value()
            if not self._use_dual
            else (
                x.value()[0] / y.value()[0],
                (x.value()[1] * y.value()[0] - x.value()[0] * y.value()[1])
                / y.value()[0] ** 2,
            )
        ),
    )


def RDiv(self, other):
    """Compute n2 / n1 for nodes n1, n2"""
    other = self._tonode(other)
    other._use_dual = self._use_dual
    return binary(
        self,
        other,
        "{}\\{}".format(self._name_no_id(), other._name_no_id()),
        (
            lambda x, y: y.value() / x.value()
            if not self._use_dual
            else (
                y.value()[0] / x.value()[0],
                (y.value()[1] * x.value()[0] - y.value()[0] * x.value()[1])
                / x.value()[0] ** 2,
            )
        ),
    )


def Pow(self, other):
    """Compute n1 ^ n2 for nodes n1, n2"""
    other = self._tonode(other)
    other._use_dual = self._use_dual
    return binary(
        self,
        other,
        "{}^{}".format(self._name_no_id(), other._name_no_id()),
        (
            lambda x, y: x.value() ** y.value()
            if not self._use_dual
            else (
                x.value()[0] ** y.value(),
                y.value() * x.value()[1] * x.value()[0] ** (y.value() - 1),
            )
        ),
    )


def Mod(self, other):
    """Compute n1 % n2 for nodes n1, n2"""
    if self._use_dual:
        raise NotImplementedError("Not Implemented!")
    other = self._tonode(other)
    return binary(
        self,
        other,
        "{}%{}".format(self._name_no_id(), other._name_no_id()),
        (lambda x, y: x.value() % y.value()),
    )


def Negate(self):
    """Compute -1 * n for node n"""
    return unary(
        self,
        "(-{})".format(self._name_no_id()),
        (
            lambda x: -self.value()
            if not self._use_dual
            else (-1 * self.value()[0], -1 * self.value()[1])
        ),
    )


def Invert(self):
    """Compute 1 / n for node n"""
    return unary(
        self,
        "1/{}".format(self._name_no_id()),
        (
            lambda x: 1 / self.value()
            if not self._use_dual
            else (1 / self.value()[0], -self.value()[1] / (self.value()[0] ** 2))
        ),
    )


def Sum(self, *others):
    """Compute sum(n1, n2, ....) for nodes n1, n2, ..."""
    others_nodes = []
    for other in others:
        if isinstance(other, Node):
            if other._use_dual != self._use_dual:
                raise NotImplementedError("Not Implemented!")
        x = self._tonode(other)
        x._use_dual = self._use_dual
        others_nodes.append(x)

    return n_nary(
        self,
        others_nodes,
        "Sum({},{})".format(
            self._name_no_id(), ",".join(other._name_no_id() for other in others_nodes)
        ),
        (
            lambda *args: sum(x.value() for x in args)
            if not self._use_dual
            else (sum([x.value()[0] for x in args]), sum(x.value()[1] for x in args))
        ),
    )


def Average(self, *others):
    """Compute mean(n1, n2, ....) for nodes n1, n2, ..."""
    others_nodes = []
    for other in others:
        if isinstance(other, Node):
            if other._use_dual != self._use_dual:
                raise NotImplementedError("Not Implemented!")
        x = self._tonode(other)
        x._use_dual = self._use_dual
        others_nodes.append(x)

    return n_nary(
        self,
        others_nodes,
        "Average({},{})".format(
            self._name_no_id(), ",".join(other._name_no_id() for other in others_nodes)
        ),
        (
            lambda *args: sum(x.value() for x in args) / len(args)
            if not self._use_dual
            else (
                (
                    sum([x.value()[0] for x in args]) / len(args),
                    sum(x.value()[1] for x in args) / len(args),
                )
            )
        ),
    )


Mean = Average

#####################
# Logical Operators #
#####################


def Or(self, other):
    """Compute n1 or n2 for nodes n1, n2"""
    other = self._tonode(other)
    other._use_dual = self._use_dual
    return binary(
        self,
        other,
        "{}||{}".format(self._name_no_id(), other._name_no_id()),
        (lambda x, y: x.value() or y.value()),
    )


def And(self, other):
    """Compute n1 and n2 for nodes n1, n2"""
    other = self._tonode(other)
    other._use_dual = self._use_dual
    return binary(
        self,
        other,
        "{}&&{}".format(self._name_no_id(), other._name_no_id()),
        (lambda x, y: x.value() and y.value()),
    )


def Not(self):
    """Compute not n for node n"""
    return unary(self, "!{}".format(self._name_no_id()), (lambda x: not x.value()))


##########################
# Mathematical Functions #
##########################
def Sin(self):
    """Compute sin(n) for node n"""
    return unary(
        self,
        "sin({})".format(self._name_no_id()),
        (
            lambda x: math.sin(self.value())
            if not self._use_dual
            else (
                math.sin(self.value()[0]),
                math.cos(self.value()[0]) * self.value()[1],
            )
        ),
    )


def Cos(self):
    """Compute cos(n) for node n"""
    return unary(
        self,
        "cos({})".format(self._name_no_id()),
        (
            lambda x: math.cos(self.value())
            if not self._use_dual
            else (
                math.cos(self.value()[0]),
                -1 * math.sin(self.value()[0]) * self.value()[1],
            )
        ),
    )


def Tan(self):
    """Compute tan(n) for node n"""
    return unary(
        self,
        "tan({})".format(self._name_no_id()),
        (
            lambda x: math.tan(self.value())
            if not self._use_dual
            else (
                math.tan(self.value()[0]),
                self.value()[1] * (1 / math.cos(self.value()[0])) ** 2,
            )
        ),
    )


def Arcsin(self):
    """Compute arcsin(n) for node n"""
    return unary(
        self,
        "arcsin({})".format(self._name_no_id()),
        (
            lambda x: math.asin(self.value())
            if not self._use_dual
            else (
                math.asin(self.value()[0]),
                self.value()[1] / math.sqrt(1 - self.value()[0] ** 2),
            )
        ),
    )


def Arccos(self):
    """Compute arccos(n) for node n"""
    return unary(
        self,
        "arccos({})".format(self._name_no_id()),
        (
            lambda x: math.acos(self.value())
            if not self._use_dual
            else (
                math.acos(self.value()[0]),
                -1 * self.value()[1] / math.sqrt(1 - self.value()[0] ** 2),
            )
        ),
    )


def Arctan(self):
    """Compute arctan(n) for node n"""
    return unary(
        self,
        "arctan({})".format(self._name_no_id()),
        (
            lambda x: math.atan(self.value())
            if not self._use_dual
            else (
                math.atan(self.value()[0]),
                self.value()[1] / (1 + self.value()[0] ** 2),
            )
        ),
    )


def Abs(self):
    """Compute abs(n) for node n"""
    return unary(
        self,
        "||{}||".format(self._name_no_id()),
        (
            lambda x: abs(self.value())
            if not self._use_dual
            else (
                abs(self.value()[0]),
                self.value()[1] * self.value()[0] / abs(self.value()[0]),
            )
        ),
    )


def Sqrt(self):
    """Compute sqrt(n) for node n"""
    return unary(
        self,
        "sqrt({})".format(self._name_no_id()),
        (
            lambda x: math.sqrt(self.value())
            if not self._use_dual
            else (
                math.sqrt(self.value()[0]),
                self.value()[1] * 0.5 / math.sqrt(self.value()[0]),
            )
        ),
    )


def Log(self):
    """Compute log(n) for node n"""
    return unary(
        self,
        "log({})".format(self._name_no_id()),
        (
            lambda x: math.log(self.value())
            if not self._use_dual
            else (math.log(self.value()[0]), self.value()[1] / self.value()[0])
        ),
    )


def Exp(self):
    """Compute exp(n) for node n"""
    return unary(
        self,
        "exp({})".format(self._name_no_id()),
        (
            lambda x: math.exp(self.value())
            if not self._use_dual
            else (
                math.exp(self.value()[0]),
                self.value()[1] * math.exp(self.value()[0]),
            )
        ),
    )


def Erf(self):
    """Compute erf(n) for node n"""
    return unary(
        self,
        "erf({})".format(self._name_no_id()),
        (
            lambda x: math.erf(self.value())
            if not self._use_dual
            else (
                math.erf(self.value()[0]),
                self.value()[1]
                * (2 / math.sqrt(math.pi))
                * math.exp(-1 * math.pow(self.value()[0], 2)),
            )
        ),
    )


##############
# Converters #
##############
def Float(self):
    """Compute float(n) for node n"""
    return unary(
        self,
        "float({})".format(self._name_no_id()),
        (
            lambda x: float(self.value())
            if not self._use_dual
            else float(self.value()[0])
        ),
    )


def Int(self):
    """Compute int(n) for node n"""
    return unary(
        self,
        "int({})".format(self._name_no_id()),
        (lambda x: int(self.value()) if not self._use_dual else int(self.value()[0])),
    )


def Bool(self):
    """Compute bool(n) for node n"""
    return unary(
        self,
        "bool({})".format(self._name_no_id()),
        (lambda x: bool(self.value()) if not self._use_dual else bool(self.value()[0])),
    )


def __Bool__(self):
    if self.value() is None:
        return False
    return bool(self.value())


def Str(self):
    """Compute str(n) for node n"""
    return unary(
        self,
        "str({})".format(self._name_no_id()),
        (
            lambda x: str(self.value())
            if not self._use_dual
            else str(self.value()[0]) + "+" + str(self.value()[1]) + "ε"
        ),
    )


###################
# Python Builtins #
###################
def Floor(self):
    """Compute floor(n) for node n"""
    return unary(
        self,
        "floor({})".format(self._name_no_id()),
        (
            lambda x: math.floor(self.value())
            if not self._use_dual
            else (math.floor(self.value()[0]), math.floor(self.value()[1]))
        ),
    )


def Ceil(self):
    """Compute ceil(n) for node n"""
    return unary(
        self,
        "ceil({})".format(self._name_no_id()),
        (
            lambda x: math.ceil(self.value())
            if not self._use_dual
            else (math.ceil(self.value()[0]), math.ceil(self.value()[1]))
        ),
    )


def Round(self, ndigits=0):
    """Compute round(n, ndigits) for node n"""
    return unary(
        self,
        "round({}, {})".format(self._name_no_id(), ndigits),
        (
            lambda x: round(self.value(), ndigits=ndigits)
            if not self._use_dual
            else (
                round(self.value()[0], ndigits=ndigits),
                round(self.value()[1], ndigits=ndigits),
            )
        ),
    )


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
def Equal(self, other):
    """Compute n1 == n2 for nodes n1 and n2"""
    if isinstance(other, Node) and super(Node, self).__eq__(other):
        return True
    other = self._tonode(other)
    other._use_dual = self._use_dual
    return binary(
        self,
        other,
        self._name_no_id() + "==" + other._name_no_id(),
        (
            lambda x, y: x.value() == y.value()
            if not self._use_dual
            else x.value()[0] == y.value()[0]
        ),
    )


def NotEqual(self, other):
    """Compute n1 != n2 for nodes n1 and n2"""
    if isinstance(other, Node) and super(Node, self).__eq__(other):
        return False
    other = self._tonode(other)
    other._use_dual = self._use_dual
    return binary(
        self,
        other,
        self._name_no_id() + "!=" + other._name_no_id(),
        (
            lambda x, y: x.value() != y.value()
            if not self._use_dual
            else x.value()[0] != y.value()[0]
        ),
    )


def Ge(self, other):
    """Compute n1 >= n2 for nodes n1 and n2"""
    if isinstance(other, Node) and super(Node, self).__eq__(other):
        return True
    other = self._tonode(other)
    other._use_dual = self._use_dual
    return binary(
        self,
        other,
        self._name_no_id() + ">=" + other._name_no_id(),
        (
            lambda x, y: x.value() >= y.value()
            if not self._use_dual
            else x.value()[0] > y.value()[0] or x.value()[0] == y.value()[0]
        ),
    )


def Gt(self, other):
    """Compute n1 > n2 for nodes n1 and n2"""
    if isinstance(other, Node) and super(Node, self).__eq__(other):
        return False
    other = self._tonode(other)
    other._use_dual = self._use_dual
    return binary(
        self,
        other,
        self._name_no_id() + ">" + other._name_no_id(),
        (
            lambda x, y: x.value() > y.value()
            if not self._use_dual
            else x.value()[0] > y.value()[0]
        ),
    )


def Le(self, other):
    """Compute n1 <= n2 for nodes n1 and n2"""
    if isinstance(other, Node) and super(Node, self).__eq__(other):
        return True
    other = self._tonode(other)
    other._use_dual = self._use_dual
    return binary(
        self,
        other,
        self._name_no_id() + "<=" + other._name_no_id(),
        (
            lambda x, y: x.value() <= y.value()
            if not self._use_dual
            else x.value()[0] <= y.value()[0] or x.value()[0] == y.value()[0]
        ),
    )


def Lt(self, other):
    """Compute n1 < n2 for nodes n1 and n2"""
    if isinstance(other, Node) and super(Node, self).__eq__(other):
        return False
    other = self._tonode(other)
    other._use_dual = self._use_dual
    return binary(
        self,
        other,
        self._name_no_id() + "<" + other._name_no_id(),
        (
            lambda x, y: x.value() < y.value()
            if not self._use_dual
            else x.value()[0] < y.value()[0]
        ),
    )


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
# Node.__trunc__ = Len
Node.round = Round
Node.__round__ = Round
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
