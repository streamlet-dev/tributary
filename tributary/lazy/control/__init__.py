
from .utils import _CONTROL_GRAPHVIZSHAPE
from ..base import Node


def If(if_node, satisfied_node, unsatisfied_node=None, *elseifs):
    '''Node to return satisfied if if_node else unsatisfied

    Args:
        if_node (Node): input booleans
        satisfied_node (Node): input to return if true
        unsatisfied_node (Node): input to return if False
        elseifs (Tuple(Node)): input of boolean/value pairs to evaluate as else if statements
    '''
    if len(elseifs) % 2 != 0:
        raise Exception('Else ifs must be in pairs')

    def foo():
        if if_node():
            return satisfied_node()
        return unsatisfied_node() if unsatisfied_node else None

    if isinstance(if_node._self_reference, Node):
        return if_node._gennode("If",
                                foo,
                                [],
                                graphvizshare=_CONTROL_GRAPHVIZSHAPE)
    return if_node._gennode("If",
                            foo,
                            [],
                            graphvizshare=_CONTROL_GRAPHVIZSHAPE)


Node.if_ = If
