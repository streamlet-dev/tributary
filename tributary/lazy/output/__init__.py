from ..node import Node


def _print(node, counter=0, cache=None):
    if cache is None:
        cache = {}

    key = cache.get(id(node), str(node) + ' (#' + str(counter) + ')')
    cache[id(node)] = key

    if node._dirty or node._subtree_dirty() or node._always_dirty:
        key += '(dirty)'

    ret = {key: []}
    counter += 1

    if node._dependencies:
        for call, deps in node._dependencies.items():
            # callable node
            if hasattr(call, '_node_wrapper') and \
                    call._node_wrapper is not None:
                val, counter = call._node_wrapper._print(counter, cache)
                ret[key].append(val)

            # args
            for arg in deps[0]:
                val, counter = arg._print(counter, cache)
                ret[key].append(val)

            # kwargs
            for kwarg in deps[1].values():
                val, counter = kwarg._print(counter, cache)
                ret[key].append(val)

    return ret, counter


def Print(node):
    return node._print(0, {})[0]


def Graph(node):
    return node.print()


def GraphViz(node):
    d = node.graph()

    from graphviz import Digraph
    dot = Digraph(node._name, strict=True)
    dot.format = 'png'

    def rec(nodes, parent):
        for d in nodes:
            if not isinstance(d, dict):
                if '(dirty)' in d:
                    dot.node(d.replace('(dirty)', ''), color='red')
                    dot.edge(d.replace('(dirty)', ''), parent.replace('(dirty)', ''), color='red')
                else:
                    dot.node(d)
                    dot.edge(d, parent.replace('(dirty)', ''))
            else:
                for k in d:
                    if '(dirty)' in k:
                        dot.node(k.replace('(dirty)', ''), color='red')
                        rec(d[k], k)
                        dot.edge(k.replace('(dirty)', ''), parent.replace('(dirty)', ''), color='red')
                    else:
                        dot.node(k)
                        rec(d[k], k)
                        dot.edge(k, parent.replace('(dirty)', ''))

    for k in d:
        if '(dirty)' in k:
            dot.node(k.replace('(dirty)', ''), color='red')
        else:
            dot.node(k)
        rec(d[k], k)
    return dot


Node._print = _print
Node.print = Print
Node.graph = Graph
Node.graphviz = GraphViz
