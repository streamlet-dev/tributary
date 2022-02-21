from ..node import Node


def Graph(node):
    if not node.upstream():
        # leaf node
        return {node: []}
    return {node: [_.graph() for _ in node.upstream()]}


def Print(node, level=0):
    ret = "    " * (level - 1) if level else ""

    if not node.upstream():
        # leaf node
        return ret + "  \\  " + repr(node)
    return (
        "    " * level
        + repr(node)
        + "\n"
        + "\n".join(_.print(level + 1) for _ in node.upstream())
    )


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


Node.print = Print
Node.graph = Graph
Node.graphviz = GraphViz
Node.dagre = Dagre
