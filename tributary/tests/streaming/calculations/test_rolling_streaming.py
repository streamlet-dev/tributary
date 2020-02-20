import tributary.streaming as ts


def foo():
    yield 1
    yield 1
    yield 1
    yield 1
    yield 1


def foo2():
    yield 1
    yield 2
    yield 0
    yield 5
    yield 4


def foo3():
    yield 1
    yield 2
    yield 3
    yield 4
    yield 5


class TestRolling:
    def test_count(self):
        assert ts.run(ts.RollingCount(ts.Foo(foo))) == [1, 2, 3, 4, 5]

    def test_sum(self):
        assert ts.run(ts.RollingSum(ts.Foo(foo))) == [1, 2, 3, 4, 5]

    def test_min(self):
        assert ts.run(ts.RollingMin(ts.Foo(foo2))) == [1, 1, 0, 0, 0]

    def test_max(self):
        assert ts.run(ts.RollingMax(ts.Foo(foo2))) == [1, 2, 2, 5, 5]

    def test_average(self):
        assert ts.run(ts.RollingAverage(ts.Foo(foo3))) == [1, 1.5, 2, 2.5, 3]
