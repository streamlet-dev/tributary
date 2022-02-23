import ast
import inspect
import textwrap

import tributary.lazy as t


blerg = t.Node(value=5)


class Func5(t.LazyGraph):
    def __init__(self):
        super().__init__()
        self.x = self.node(name="x", value=None)

    def reset(self):
        self.x = None

    @t.node()
    def z(self):
        return self.x() | self.y() | blerg

    def zz(self):
        return self.x | self.y() | blerg

    @t.node()
    def y(self):
        return 10


source = inspect.getsource(Func5.zz)
root = ast.parse(textwrap.dedent(source)).body[0]

print(ast.dump(root, indent=4))


def isClassAttribute(node):
    # self.aNode()
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.value.id == "self"
    ):
        return node.func.attr
    # self.aNode
    elif isinstance(node, ast.Attribute) and node.value.id == "self":
        return node.attr


def getClassAttributesUsedInMethod(root):
    attribute_deps = []
    for node in ast.walk(root):
        attr = isClassAttribute(node)
        if attr:
            attribute_deps.append(attr)
    return attribute_deps


ads = getClassAttributesUsedInMethod(root)
# print("attribute deps:", ads)
# print("***Args***")
# print(root.args.posonlyargs)
# print(root.args.args)
# print(root.args.kwonlyargs)
# print(root.args.kw_defaults)
# print(root.args.defaults)
# print("***")


# append attribute args to method definition
def addAttributeDepsToMethodSignature(root, attribute_deps):
    for attribute_dep in attribute_deps:
        append = True

        for meth_arg in root.args.args:
            if meth_arg.arg == attribute_dep:
                append = False

        if append:
            root.args.args.append(ast.arg(attribute_dep))


addAttributeDepsToMethodSignature(root, ads)


class Transformer(ast.NodeTransformer):
    def generic_visit(self, node):
        # Need to call super() in any case to visit child nodes of the current one.
        super().generic_visit(node)

        # if isClassAttribute(node):
        #     return ast.Attribute()
        # Attribute(
        #                 value=Name(id='self', ctx=Load()),
        #                 attr='x',
        #                 ctx=Load()),
        return node


print(ast.unparse(Transformer().visit(root)))


# names = sorted({node.id for node in ast.walk(root) if isinstance(node, ast.Name)})
# print(names)

# f = Func5()
# print(f.z.graph())
# print(f.z())
# # assert f.z()() == 10
# # assert f.x() is None

# # f.x = 5

# # assert f.x() == 5
# # assert f.z()() == 5

# # f.reset()

# # assert f.x() is None
# # assert f.z()() == 10
