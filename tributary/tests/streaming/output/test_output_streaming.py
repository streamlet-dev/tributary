import time
import tributary.streaming as ts


class TestOutput:
    def setup(self):
        time.sleep(0.5)

    def test_print(self):
        val = 0

        def func():
            nonlocal val
            val += 1
            return val

        assert ts.run(ts.Print(ts.Timer(func, count=2))) == [1, 2]

    def test_print_generator(self):
        def func():
            yield 1
            yield 2
            yield 3
            yield 4
            yield 5

        assert ts.run(ts.Print(ts.Timer(func, count=2))) == [1, 2]

    def test_graphviz(self):
        def func():
            yield 1

        assert ts.GraphViz(ts.Print(ts.Timer(func, count=2)))

    def test_dagre(self):
        def func():
            yield 1

        assert ts.Dagre(ts.Print(ts.Timer(func, count=2)))
