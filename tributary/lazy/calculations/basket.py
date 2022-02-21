from .ops import unary
from ..node import Node


def Len(self):
    """Compute len(n) for node n"""
    return unary(self, "len({})".format(self._name_no_id), (lambda x: len(x)))


def CountBasket(self):
    """Compute len(n) for node n"""
    return unary(self, "count({})".format(self._name_no_id), (lambda x: len(x)))


def MaxBasket(self):
    """Compute max(n) for node n"""
    return unary(self, "max({})".format(self._name_no_id), (lambda x: max(x)))


def MinBasket(self):
    """Compute max(n) for node n"""
    return unary(self, "min({})".format(self._name_no_id), (lambda x: min(x)))


def SumBasket(self):
    """Compute sum(n) for node n"""
    return unary(self, "sum({})".format(self._name_no_id), (lambda x: sum(x)))


def AverageBasket(self):
    """Compute mean(n) for node n"""
    return unary(
        self,
        "average({})".format(self._name_no_id),
        (lambda x: sum(x) / len(x)),
    )


MeanBasket = AverageBasket

Node.__len__ = Len
Node.len = Len
Node.countBasket = CountBasket
Node.minBasket = MinBasket
Node.maxBasket = MaxBasket
Node.sumBasket = SumBasket
Node.averageBasket = AverageBasket
Node.meanBasket = MeanBasket
