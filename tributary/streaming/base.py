

class StreamingGraph(object):
    def __init__(self, output_node):
        self._node = output_node

    def graph(self):
        return self._node.graph()

    def graphviz(self):
        return self._node.graphviz()

    def dagre(self):
        return self._node.dagre()

    def run(self):
        from tributary.streaming import run
        return run(self._node)
