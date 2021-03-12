from ..node import Node


def _print(node, cache=None):
    if cache is None:
        cache = {}

    if id(node) in cache:
        # loop, return None
        return node

    cache[id(node)] = node

    ret = {node: []}

    if node._dependencies:
        for call, deps in node._dependencies.items():
            # callable node
            if hasattr(call, "_node_wrapper") and call._node_wrapper is not None:
                ret[node].append(call._node_wrapper._print(cache))

            # args
            for arg in deps[0]:
                ret[node].append(arg._print(cache))

            # kwargs
            for kwarg in deps[1].values():
                ret[node].append(kwarg._print(cache))

    return ret


def Print(node):
    return node._print({})


def Graph(node):
    if isinstance(node, Node):
        return node.print()

    ret = {}
    for n in node:
        ret.update(n.print())
    return ret


def GraphViz(node):
    # allow for lists of nodes
    if isinstance(node, Node):
        d = node.graph()
        name = node._name
    else:
        d = {}
        for n in node:
            d.update(n.graph())
        name = ",".join(n._name for n in node)

    from graphviz import Digraph

    dot = Digraph(name, strict=True)
    dot.format = "png"

    def rec(nodes, parent):
        for d in nodes:
            if not isinstance(d, dict):
                if d.isDynamic():
                    dot.node(d._name, color="teal", shape=d._graphvizshape)
                    dot.edge(d._name, parent._name, color="magenta")
                elif d.isDirty():
                    dot.node(d._name, color="red", shape=d._graphvizshape)
                    dot.edge(d._name, parent._name, color="red")
                else:
                    dot.node(d._name, shape=d._graphvizshape)
                    dot.edge(d._name, parent._name)
            else:
                for k in d:
                    if k.isDynamic():
                        dot.node(k._name, color="teal", shape=k._graphvizshape)
                        rec(d[k], k)
                        dot.edge(k._name, parent._name, color="magenta")
                    elif k.isDirty():
                        dot.node(k._name, color="red", shape=k._graphvizshape)
                        rec(d[k], k)
                        dot.edge(k._name, parent._name, color="red")
                    else:
                        dot.node(k._name, shape=k._graphvizshape)
                        rec(d[k], k)
                        dot.edge(k._name, parent._name)

    for k in d:
        if k.isDynamic():
            dot.node(k._name, color="teal", shape=k._graphvizshape)
        elif k.isDirty():
            dot.node(k._name, color="red", shape=k._graphvizshape)
        else:
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
                if d.isDynamic():
                    G.setNode(
                        d._name,
                        style="fill: #0ff",
                        shape="rect" if d._graphvizshape == "box" else d._graphvizshape,
                    )
                elif d.isDirty():
                    G.setNode(
                        d._name,
                        style="fill: #f00",
                        shape="rect" if d._graphvizshape == "box" else d._graphvizshape,
                    )
                    # G.setEdge(d._name, parent._name, style='stroke: #f00')
                else:
                    G.setNode(
                        d._name,
                        style="fill: #fff",
                        shape="rect" if d._graphvizshape == "box" else d._graphvizshape,
                    )

                G.setEdge(d._name, parent._name, style="stroke: #000")
            else:
                for k in d:
                    k._dd3g = G
                    if k.isDynamic():
                        G.setNode(
                            k._name,
                            style="fill: #0ff",
                            shape="rect"
                            if k._graphvizshape == "box"
                            else k._graphvizshape,
                        )
                    elif k.isDirty():
                        G.setNode(
                            k._name,
                            style="fill: #f00",
                            shape="rect"
                            if k._graphvizshape == "box"
                            else k._graphvizshape,
                        )
                    else:
                        G.setNode(
                            k._name,
                            style="fill: #fff",
                            shape="rect"
                            if k._graphvizshape == "box"
                            else k._graphvizshape,
                        )
                    rec(d[k], k)

                    G.setEdge(k._name, parent._name, style="stroke: #000")

    for k in d:
        k._dd3g = G
        if k.isDynamic():
            G.setNode(
                k._name,
                style="fill: #0ff",
                shape="rect" if k._graphvizshape == "box" else k._graphvizshape,
            )
        elif k.isDirty():
            G.setNode(
                k._name,
                style="fill: #f00",
                shape="rect" if k._graphvizshape == "box" else k._graphvizshape,
            )
        else:
            G.setNode(
                k._name,
                style="fill: #fff",
                shape="rect" if k._graphvizshape == "box" else k._graphvizshape,
            )
        rec(d[k], k)

    graph = dd3.DagreD3Widget(graph=G)
    return graph


Node._print = _print
Node.print = Print
Node.graph = Graph
Node.graphviz = GraphViz
Node.dagre = Dagre
