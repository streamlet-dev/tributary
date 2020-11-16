import time
import tributary.lazy as tl
import tributary.streaming as ts
import tributary as t


class TestLazyToStreaming:
    def setup(self):
        time.sleep(0.5)

    def test_function(self):
        def foo(*args):
            for _ in range(5):
                yield _

        lazy_node = tl.Node(callable=foo) + 5
        # 5 6 7 8 9

        streaming_node = ts.Print(ts.Foo(foo) - 5, "streaming:")
        # -5 -4 -3 -2 -1

        out = ts.Print(t.LazyToStreaming(lazy_node), "lazy:") + streaming_node
        x = ts.run(out)

        # 0 2 4 6 8
        print(x)
        assert x == [0, 2, 4, 6, 8]

    def test_function_order(self):
        def foo(*args):
            for _ in range(5):
                yield _

        lazy_node = tl.Node(callable=foo) + 5
        # 5 6 7 8 9

        streaming_node = ts.Print(ts.Foo(foo) - 5, "streaming:")
        # -5 -4 -3 -2 -1

        out = streaming_node + ts.Print(lazy_node, "lazy:")
        x = ts.run(out)

        # 0 2 4 6 8
        print(x)
        assert x == [0, 2, 4, 6, 8]

    def test_value(self):
        def foo(*args):
            for _ in range(5):
                lazy_node.setValue(_ + 1)
                yield _

        lazy_node = tl.Node(value=0)
        # 5 6 7 8 9

        streaming_node = ts.Print(ts.Foo(foo) - 5, "streaming:")
        # -5 -4 -3 -2 -1

        out = ts.Print(t.LazyToStreaming(lazy_node) + 5, "lazy:") + streaming_node
        x = ts.run(out)
        # 0 2 4 6 8
        print(x)
        assert x == [0, 2, 4, 6, 8]

    def test_value_order(self):
        lazy_node = tl.Node(value=0)
        # 5 6 7 8 9

        def foo(lazy_node=lazy_node):
            for _ in range(5):
                yield _
                lazy_node.setValue(_ + 1)

        streaming_node = ts.Print(ts.Foo(foo) - 5, "streaming:")
        # -5 -4 -3 -2 -1

        out = streaming_node + ts.Print(lazy_node + 5, "lazy:")
        x = ts.run(out)
        # 0 2 4 6 8
        print(x)
        assert x == [0, 2, 4, 6, 8]
