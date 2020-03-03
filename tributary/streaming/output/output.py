from IPython.display import display
from ..base import Node

_OUTPUT_GRAPHVIZSHAPE = "box"


def Print(node, text=''):
    def foo(val):
        print(text + str(val))
        return val

    ret = Node(foo=foo, foo_kwargs=None, name='Print', inputs=1, graphvizshape=_OUTPUT_GRAPHVIZSHAPE)

    node._downstream.append((ret, 0))
    ret._upstream.append(node)
    return ret


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
                dot.node(d, shape=d._graphvizshape)
                dot.edge(d, parent)

            else:
                for k in d:
                    dot.node(k._name, shape=k._graphvizshape)
                    rec(d[k], k)
                    dot.edge(k._name, parent._name)

    for k in d:
        dot.node(k._name, shape=k._graphvizshape)
        rec(d[k], k)

    return dot


def Dagre(node):
    import ipydagred3 as dd3
    G = dd3.Graph()
    d = Graph(node)

    def rec(nodes, parent):
        for d in nodes:
            if not isinstance(d, dict):
                d._dd3g = G
                G.setNode(d._name, shape="rect" if d._graphvizshape == "box" else d._graphvizshape)
                G.setEdge(d._name, parent)
            else:
                for k in d:
                    k._dd3g = G
                    G.setNode(k._name, shape="rect" if k._graphvizshape == "box" else k._graphvizshape)
                    G.setEdge(k._name, parent._name)
                    rec(d[k], k)

    for k in d:
        k._dd3g = G
        G.setNode(k._name, shape="rect" if k._graphvizshape == "box" else k._graphvizshape)
        rec(d[k], k)

    graph = dd3.DagreD3Widget(graph=G)
    return graph


def Perspective(node, text='', **psp_kwargs):
    psp_kwargs = psp_kwargs or {}
    from perspective import PerspectiveWidget
    p = PerspectiveWidget(psp_kwargs.pop('schema', []), **psp_kwargs)

    def foo(val):
        p.update(val)
        return val

    ret = Node(foo=foo, foo_kwargs=None, name='Perspective', inputs=1, graphvizshape=_OUTPUT_GRAPHVIZSHAPE)

    display(p)
    node._downstream.append((ret, 0))
    ret._upstream.append(node)
    return ret


Node.graph = Graph
Node.pprint = PPrint
Node.graphviz = GraphViz
Node.dagre = Dagre
Node.print = Print
Node.perspective = Perspective
