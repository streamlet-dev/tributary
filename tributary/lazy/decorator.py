from .node import Node
from ..utils import _either_type
from ..parser import (
    pprintCode,
    pprintAst,
    parseASTForMethod,
    getClassAttributesUsedInMethod,
    addAttributeDepsToMethodSignature,
    Transformer,
)


@_either_type
def node(meth, dynamic=False):
    """Convert a method into a lazy node"""

    # parse out function
    root, mod = parseASTForMethod(meth)

    # grab accessed attributes
    ads = getClassAttributesUsedInMethod(root)

    # # add them to signature
    # addAttributeDepsToMethodSignature(root, ads)

    # # modify
    # new_root = Transformer().visit(root)

    # # remove decorators
    # new_root.decorator_list = []

    # # replace in module
    # mod.body[0] = new_root

    # # pprintAst(new_root)
    # # pprintCode(mod)
    # meth = exec(compile(mod, meth.__code__.co_filename, "exec"), meth.__globals__)
    # print(type(meth))
    # import ipdb; ipdb.set_trace()

    new_node = Node(value=meth)
    new_node._nodes_to_bind = ads

    # if new_node._callable_is_method:
    #     ret = lambda self, *args, **kwargs: new_node._bind(  # noqa: E731
    #         self, *args, **kwargs
    #     )
    # else:
    #     ret = lambda *args, **kwargs: new_node._bind(  # noqa: E731
    #         None, *args, **kwargs
    #     )
    # ret._node_wrapper = new_node
    # return ret

    return new_node