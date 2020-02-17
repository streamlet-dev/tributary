from IPython.display import display
from ..base import Node


class Print(Node):
    def __init__(self, node, text=''):
        def foo(val):
            return val
        super().__init__(foo=foo, foo_kwargs=None, name='Print', inputs=1)

        node._downstream.append((self, 0))
        self._upstream.append(node)


def Graph(node):
    if not node._upstream:
        # leaf node
        return {node: []}
    return {node: [_.graph() for _ in node._upstream]}


def PPrint(node, level=0):
    ret = '    ' * (level - 1) if level else ''

    if not node._upstream:
        # leaf node
        return ret + '  \\  ' + repr(node)
    return '    ' * level + repr(node) + '\n' + '\n'.join(_.pprint(level + 1) for _ in node._upstream)


def GraphViz(node):
    d = Graph(node)

    from graphviz import Digraph
    dot = Digraph("Graph", strict=False)
    dot.format = 'png'

    def rec(nodes, parent):
        for d in nodes:
            if not isinstance(d, dict):
                dot.node(d)
                dot.edge(d, parent)

            else:
                for k in d:
                    dot.node(k._name)
                    rec(d[k], k)
                    dot.edge(k._name, parent._name)

    for k in d:
        dot.node(k._name)
        rec(d[k], k)

    return dot


class Perspective(Node):
    def __init__(self, node, text='', psp_kwargs=None):
        psp_kwargs = psp_kwargs or {}
        from perspective import PerspectiveWidget
        p = PerspectiveWidget(psp_kwargs.pop('schema', []), **psp_kwargs)

        def foo(val):
            p.update(val)
            return val
        super().__init__(foo=foo, foo_kwargs=None, name='Print', inputs=1)

        display(p)
        node._downstream.append((self, 0))
        self._upstream.append(node)
        self._name = "Perspective"


Node.graph = Graph
Node.pprint = PPrint
Node.graphviz = GraphViz
Node.print = Print
Node.perspective = Perspective
