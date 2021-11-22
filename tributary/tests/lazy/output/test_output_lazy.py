import tributary.lazy as tl


class TestOutput:
    def test_graphviz(self):
        def foo():
            return 1

        assert tl.GraphViz(tl.Node(callable=foo, always_dirty=True))

    def test_dagre(self):
        def foo():
            return 1

        assert tl.Dagre(tl.Node(callable=foo, always_dirty=True))

    def test_multiple_graph(self):
        n = tl.Node(5)
        n1 = n + 5
        n2 = n + 4
        assert tl.Graph([n1, n2]) is not None
