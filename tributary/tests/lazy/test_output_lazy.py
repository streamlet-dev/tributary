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
