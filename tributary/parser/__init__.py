import ast
import inspect
import textwrap


def parseASTForMethod(meth):
    source = inspect.getsource(meth)
    dedented_source = textwrap.dedent(source)
    mod = ast.parse(dedented_source)
    return mod.body[0], mod


def pprintAst(astNode):
    print(ast.dump(astNode, indent=4))


def pprintCode(astNode):
    print(ast.unparse(astNode))


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
    attribute_deps = set()
    for node in ast.walk(root):
        attr = isClassAttribute(node)
        if attr:
            attribute_deps.add(attr)
    return attribute_deps


# append attribute args to method definition
def addAttributeDepsToMethodSignature(root, attribute_deps):
    for attribute_dep in attribute_deps:
        append = True

        for meth_arg in root.args.args:
            if meth_arg.arg == attribute_dep:
                append = False

        if append:
            if root.args.args:
                # use first one
                lineno = root.args.args[0].lineno
            else:
                # use definition
                lineno = root.lineno

            # set lineno to as close as we can, but can't do col_offset really
            root.args.args.append(ast.arg(attribute_dep, lineno=lineno, col_offset=0))


class Transformer(ast.NodeTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._transformed_to_nodes = []

    def generic_visit(self, node):
        # Need to call super() in any case to visit child nodes of the current one.
        super().generic_visit(node)

        if isClassAttribute(node) and isinstance(node, ast.Call):
            # Call(
            #     func=Attribute(
            #         value=Name(id='self', ctx=Load()),
            #         attr='func2',
            #         ctx=Load()),
            #     args=[],
            #     keywords=[])
            #
            # then replace with argument
            #
            # Name(id='blerg', ctx=Load())
            return ast.Name(
                id=node.func.attr,
                ctx=ast.Load(),
                lineno=node.func.lineno,
                col_offset=node.func.col_offset,
            )
        elif isClassAttribute(node) and isinstance(node, ast.Attribute):
            # Attribute(
            #     value=Name(id='self', ctx=Load()),
            #     attr='func2',
            #     ctx=Load())
            #
            # then replace with argument
            #
            # Name(id='blerg', ctx=Load())
            name = ast.Name(
                id=node.attr,
                ctx=ast.Load(),
                lineno=node.lineno,
                col_offset=node.col_offset,
            )
            self._transformed_to_nodes.append(name)
            return name
        elif isinstance(node, ast.Call) and node.func in self._transformed_to_nodes:
            # if we transformed the inner attribute but its still being called, e.g.
            # def func1(self, func2):
            #    return func2() + 1

            # in this case, just promote the inner name to replace the call
            return node.func

        return node
