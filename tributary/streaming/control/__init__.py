
from .utils import _CONTROL_GRAPHVIZSHAPE
from ..base import Node


def If(if_node, satisfied_node, unsatisfied_node=None, *elseifs):
    '''Node to return satisfied if if_node else unsatisfied

    Args:
        if_node (Node): input stream of booleans
        satisfied_node (Node): input stream to return if true
        unsatisfied_node (Node): input stream to return if False
        elseifs (Tuple(Node)): input stream of boolean/value pairs to evaluate as else if statements
    '''
    if len(elseifs) % 2 != 0:
        raise Exception('Else ifs must be in pairs')

    def foo(conditional, if_val, else_val):
        if conditional:
            return if_val
        return else_val

    ret = Node(foo=foo, foo_kwargs=None, name='If', inputs=3, graphvizshape=_CONTROL_GRAPHVIZSHAPE)
    ret._count = 0
    if_node._downstream.append((ret, 0))
    satisfied_node._downstream.append((ret, 1))
    unsatisfied_node._downstream.append((ret, 2))
    ret._upstream.append(if_node)
    ret._upstream.append(satisfied_node)
    ret._upstream.append(unsatisfied_node)
    return ret


Node.if_ = If
