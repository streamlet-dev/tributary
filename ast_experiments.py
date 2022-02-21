import ast
import inspect
import textwrap

from attr import attr
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
        return self.x() | self.y() | blerg

    @t.node()
    def y(self):
        return 10

source = inspect.getsource(Func5.zz)
root = ast.parse(textwrap.dedent(source)).body[0]

print(ast.dump(root, indent=4))

attribute_deps = []
# code for parsing out self's node deps
for node in ast.walk(root):
    if isinstance(node, ast.Attribute) and node.value.id == "self":
        attribute_deps.append(node.attr)

print("attribute deps:", attribute_deps)
print("***Args***")
print(root.args.posonlyargs)
print(root.args.args)
print(root.args.kwonlyargs)
print(root.args.kw_defaults)
print(root.args.defaults)
print("***")

class Transformer(ast.NodeTransformer):
    def generic_visit(self, node):
        # Need to call super() in any case to visit child nodes of the current one.
        super().generic_visit(node)
        ordered_dict_conditions = (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == 'OrderedDict'
            and len(node.args) == 1
            and isinstance(node.args[0], ast.List)
        )
        if ordered_dict_conditions:
            return ast.Dict(
                [x.elts[0] for x in node.args[0].elts],
                [x.elts[1] for x in node.args[0].elts]
            )
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

