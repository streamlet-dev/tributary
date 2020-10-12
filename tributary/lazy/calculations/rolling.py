from ..node import Node

# def RollingCount(node):
#     def foo(node=node):
#         return node()
#     # make new node
#     ret = node._gennode('RollingCount({})'.format(node._name), foo, [node])
#     return ret


def RollingCount(node):
    raise NotImplementedError()


def RollingMax(node):
    raise NotImplementedError()


def RollingMin(node):
    raise NotImplementedError()


def RollingSum(node):
    raise NotImplementedError()


def RollingAverage(node):
    raise NotImplementedError()


def SMA(node, window_width=10, full_only=False):
    raise NotImplementedError()


def EMA(node, window_width=10, full_only=False, alpha=None, adjust=False):
    raise NotImplementedError()


def Last(node):
    raise NotImplementedError()


def First(node):
    raise NotImplementedError()


def Diff(node):
    raise NotImplementedError()


Node.rollingCount = RollingCount
Node.rollingMin = RollingMin
Node.rollingMax = RollingMax
Node.rollingSum = RollingSum
Node.rollingAverage = RollingAverage
Node.diff = Diff
Node.sma = SMA
Node.ema = EMA
Node.last = Last
Node.first = First
