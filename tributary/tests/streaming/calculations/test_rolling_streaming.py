import tributary.streaming as ts
import pandas as pd


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


def foo4():
    for _ in range(10):
        yield _


def foo5():
    yield [1, 1, 2]
    yield [1, 2, 3]
    yield [3, 4, 5]


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

    def test_diff(self):
        ret = ts.run(ts.Diff(ts.Foo(foo2)))

        vals = [1, 2, 0, 5, 4]
        comp = [None] + [vals[i] - vals[i - 1] for i in range(1, 5)]
        assert ret[0] is None
        for i, x in enumerate(ret[1:]):
            assert (x - comp[i + 1]) < 0.001

    def test_sma(self):
        ret = ts.run(ts.SMA(ts.Foo(foo4)))
        comp = pd.Series([_ for _ in range(10)]).rolling(10, min_periods=1).mean()
        for i, x in enumerate(ret):
            assert (x - comp[i]) < 0.001

    def test_ema(self):
        ret = ts.run(ts.EMA(ts.Foo(foo4)))
        comp = pd.Series([_ for _ in range(10)]).ewm(span=10, adjust=False).mean()
        for i, x in enumerate(ret):
            assert (x - comp[i]) < 0.001

    def test_ema2(self):
        ret = ts.run(ts.EMA(ts.Foo(foo4), alpha=1 / 10, adjust=True))
        comp = pd.Series([_ for _ in range(10)]).ewm(alpha=1 / 10, adjust=True).mean()
        for i, x in enumerate(ret):
            assert (x - comp[i]) < 0.001

    def test_last(self):
        assert ts.run(ts.Last(ts.Foo(foo2))) == [1, 2, 0, 5, 4]

    def test_first(self):
        assert ts.run(ts.First(ts.Foo(foo2))) == [1, 1, 1, 1, 1]

    def test_last_iter(self):
        assert ts.run(ts.Last(ts.Foo(foo5))) == [2, 3, 5]

    def test_first_iter(self):
        assert ts.run(ts.First(ts.Foo(foo5))) == [1, 1, 1]
