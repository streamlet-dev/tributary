from .utils import _CALCULATIONS_GRAPHVIZSHAPE
from ..base import Node


def Count(node):
    '''Node to count inputs

    Args:
        node (Node): input stream
    '''
    def foo(val):
        ret._count += 1
        return ret._count

    ret = Node(foo=foo, foo_kwargs=None, name='Count', inputs=1, graphvizshape=_CALCULATIONS_GRAPHVIZSHAPE)
    ret._count = 0
    node._downstream.append((ret, 0))
    ret._upstream.append(node)
    return ret


def Max(node):
    '''Node to take rolling max of inputs

    Args:
        node (Node): input stream
    '''
    def foo(val):
        ret._max = max(ret._max, val) if ret._max is not None else val
        return ret._max

    ret = Node(foo=foo, foo_kwargs=None, name='Max', inputs=1, graphvizshape=_CALCULATIONS_GRAPHVIZSHAPE)
    ret._max = None

    node._downstream.append((ret, 0))
    ret._upstream.append(node)
    return ret


def Min(node):
    '''Node to take rolling min of inputs

    Args:
        node (Node): input stream
    '''
    def foo(val):
        ret._min = min(ret._min, val) if ret._min is not None else val
        return ret._min

    ret = Node(foo=foo, foo_kwargs=None, name='Min', inputs=1, graphvizshape=_CALCULATIONS_GRAPHVIZSHAPE)
    ret._min = None
    node._downstream.append((ret, 0))
    ret._upstream.append(node)
    return ret


def Sum(node):
    '''Node to take rolling sum inputs

    If stream type is iterable, will do += sum(input). If input
    stream type is not iterable, will do += input.

    Args:
        node (Node): input stream
    '''
    def foo(val):
        try:
            # iterable, sum with sum function
            iter(val)
            ret._sum += sum(val)
        except TypeError:
            # not iterable, sum by value
            ret._sum += val
        return ret._sum

    ret = Node(foo=foo, foo_kwargs=None, name='Sum', inputs=1, graphvizshape=_CALCULATIONS_GRAPHVIZSHAPE)
    ret._sum = 0

    node._downstream.append((ret, 0))
    ret._upstream.append(node)
    return ret


def Average(node):
    '''Node to take the running average

    If stream type is iterable, will do (average + sum(input))/(count+len(input)).
    If input stream type is not iterable, will do (average + input)/count
    '''
    def foo(val):
        try:
            # iterable, sum with sum function
            iter(val)
            ret._sum += sum(val)
            ret._count += len(val)
        except TypeError:
            # not iterable, sum by value
            ret._sum += val
            ret._count += 1
        return ret._sum / ret._count if ret._count > 0 else float('nan')

    ret = Node(foo=foo, foo_kwargs=None, name='Average', inputs=1, graphvizshape=_CALCULATIONS_GRAPHVIZSHAPE)
    ret._sum = 0
    ret._count = 0

    node._downstream.append((ret, 0))
    ret._upstream.append(node)
    return ret


Node.rollingCount = Count
Node.rollingMin = Min
Node.rollingMax = Max
Node.rollingSum = Sum
Node.rollingAverage = Average
