from .utils import _CONTROL_GRAPHVIZSHAPE
from ..node import Node
from ...base import TributaryException


def If(if_node, satisfied_node, unsatisfied_node=None, *elseifs):
    """Node to return satisfied if if_node else unsatisfied

    Args:
        if_node (Node): input stream of booleans
        satisfied_node (Node): input stream to return if true
        unsatisfied_node (Node): input stream to return if False
        elseifs (Tuple(Node)): input stream of boolean/value pairs to evaluate as else if statements
    """
    if len(elseifs) % 2 != 0:
        raise TributaryException("Else ifs must be in pairs")

    def foo(conditional, if_val, else_val):
        # TODO else ifs
        if conditional:
            return if_val
        return else_val

    ret = Node(
        foo=foo,
        foo_kwargs=None,
        name="If",
        inputs=3,
        graphvizshape=_CONTROL_GRAPHVIZSHAPE,
    )
    ret.set("_count", 0)
    if_node >> ret
    satisfied_node >> ret
    unsatisfied_node >> ret
    return ret


Node.if_ = If
