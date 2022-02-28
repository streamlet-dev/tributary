import random

import tributary.lazy as t

from tributary.parser import (
    parseASTForMethod,
    pprintAst,
    getClassAttributesUsedInMethod,
    addAttributeDepsToMethodSignature,
    Transformer,
    pprintCode,
)

blerg = t.Node(value=5)


class Func4(t.LazyGraph):
    @t.node()
    def func1(self):
        return self.func2() + 1

    @t.node(dynamic=True)
    def func2(self):
        return random.random()


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
        return self.x | self.y()

    @t.node()
    def y(self):
        return 10


root, mod = parseASTForMethod(Func5.zz)
pprintAst(root)

ads = getClassAttributesUsedInMethod(root)
# print("attribute deps:", ads)
# print("***Args***")
# print(root.args.posonlyargs)
# print(root.args.args)
# print(root.args.kwonlyargs)
# print(root.args.kw_defaults)
# print(root.args.defaults)
# print("***")

addAttributeDepsToMethodSignature(root, ads)

new_root = Transformer().visit(root)
mod.body[0] = new_root
pprintCode(mod)

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
