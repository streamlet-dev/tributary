import tributary.lazy as tl


class TestOutput:
    def test_graphviz(self):
        def func():
            return 1

        assert tl.GraphViz(tl.Node(value=func, dynamic=True))

    def test_dagre(self):
        def func():
            return 1

        assert tl.Dagre(tl.Node(value=func, dynamic=True))

    def test_multiple_graph(self):
        n = tl.Node(5)
        n1 = n + 5
        n2 = n + 4
        assert tl.Graph([n1, n2]) is not None
