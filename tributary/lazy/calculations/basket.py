from .ops import unary
from ..node import Node


def Len(self):
    """Compute len(n) for node n"""
    return unary(
        self, "len({})".format(self._name_no_id()), (lambda x: len(self.value()))
    )


def CountBasket(self):
    """Compute len(n) for node n"""
    return unary(
        self, "count({})".format(self._name_no_id()), (lambda x: len(self.value()))
    )


def MaxBasket(self):
    """Compute max(n) for node n"""
    return unary(
        self, "max({})".format(self._name_no_id()), (lambda x: max(self.value()))
    )


def MinBasket(self):
    """Compute max(n) for node n"""
    return unary(
        self, "min({})".format(self._name_no_id()), (lambda x: min(self.value()))
    )


def SumBasket(self):
    """Compute sum(n) for node n"""
    return unary(
        self, "sum({})".format(self._name_no_id()), (lambda x: sum(self.value()))
    )


def AverageBasket(self):
    """Compute mean(n) for node n"""
    return unary(
        self,
        "average({})".format(self._name_no_id()),
        (lambda x: sum(self.value()) / len(self.value())),
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
