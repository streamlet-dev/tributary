from .ops import unary
from ..node import Node


Len = unary((lambda x: len(x),), name="Len")
CountBasket = unary((lambda x: len(x),), name="Count")
MaxBasket = unary((lambda x: max(x),), name="Count")
MinBasket = unary((lambda x: min(x),), name="Count")
SumBasket = unary((lambda x: sum(x),), name="Count")
AverageBasket = unary((lambda x: sum(x) / len(x),), name="Count")
MeanBasket = AverageBasket

Node.__len__ = Len
Node.len = Len
Node.countBasket = CountBasket
Node.minBasket = MinBasket
Node.maxBasket = MaxBasket
Node.sumBasket = SumBasket
Node.averageBasket = AverageBasket
Node.meanBasket = AverageBasket
