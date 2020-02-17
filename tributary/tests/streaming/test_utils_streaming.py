import time
import tributary.streaming as ts


def foo():
    yield 1
    yield 2


def foo2():
    yield [1, 2]
    yield [3, 4]


class TestUtils:
    def test_delay(self):
        out = ts.Delay(ts.Foo(foo), delay=5)
        now = time.time()
        ret = ts.run(out)
        assert time.time() - now > 5
        assert ret == [1, 2]

    def test_apply(self):
        def square(val):
            return val ** 2

        assert ts.run(ts.Apply(ts.Foo(foo), foo=square)) == [1, 4]

    def test_window_any_size(self):
        assert ts.run(ts.Window(ts.Foo(foo))) == [[1], [1, 2]]

    def test_window_fixed_size(self):
        assert ts.run(ts.Window(ts.Foo(foo), size=2)) == [[1], [1, 2]]

    def test_window_fixed_size_full_only(self):
        assert ts.run(ts.Window(ts.Foo(foo), size=2, full_only=True)) == [[1, 2]]

    def test_unroll(self):
        assert ts.run(ts.Unroll(ts.Foo(foo2))) == [1, 2, 3, 4]

    def test_merge(self):
        def foo1():
            yield 1
            yield 3

        def foo2():
            yield 2
            yield 4
            yield 6

        out = ts.Merge(ts.Print(ts.Foo(foo1)), ts.Print(ts.Foo(foo2)))
        assert ts.run(out) == [(1, 2), (3, 4)]

    def test_reduce(self):
        def foo1():
            yield 1
            yield 4

        def foo2():
            yield 2
            yield 5
            yield 7

        def foo3():
            yield 3
            yield 6
            yield 8

        out = ts.Reduce(ts.Foo(foo1), ts.Foo(foo2), ts.Foo(foo3))
        assert ts.run(out) == [(1, 2, 3), (4, 5, 6)]
