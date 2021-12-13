import tributary.streaming as ts
import pandas as pd


def func():
    yield 1
    yield 1
    yield 1
    yield 1
    yield 1


def func2():
    yield 1
    yield 2
    yield 0
    yield 5
    yield 4


def func3():
    yield 1
    yield 2
    yield 3
    yield 4
    yield 5


def func4():
    for _ in range(10):
        yield _


def func5():
    yield [1, 1, 2]
    yield [1, 2, 3]
    yield [3, 4, 5]


class TestRolling:
    def test_count(self):
        assert ts.run(ts.RollingCount(ts.Func(func))) == [1, 2, 3, 4, 5]

    def test_sum(self):
        assert ts.run(ts.RollingSum(ts.Func(func))) == [1, 2, 3, 4, 5]

    def test_min(self):
        assert ts.run(ts.RollingMin(ts.Func(func2))) == [1, 1, 0, 0, 0]

    def test_max(self):
        assert ts.run(ts.RollingMax(ts.Func(func2))) == [1, 2, 2, 5, 5]

    def test_average(self):
        assert ts.run(ts.RollingAverage(ts.Func(func3))) == [1, 1.5, 2, 2.5, 3]

    def test_diff(self):
        ret = ts.run(ts.Diff(ts.Func(func2)))

        vals = [1, 2, 0, 5, 4]
        comp = [None] + [vals[i] - vals[i - 1] for i in range(1, 5)]
        assert ret[0] is None
        for i, x in enumerate(ret[1:]):
            assert (x - comp[i + 1]) < 0.001

    def test_sma(self):
        ret = ts.run(ts.SMA(ts.Func(func4)))
        comp = pd.Series([_ for _ in range(10)]).rolling(10, min_periods=1).mean()
        for i, x in enumerate(ret):
            assert (x - comp[i]) < 0.001

    def test_ema(self):
        ret = ts.run(ts.EMA(ts.Func(func4)))
        comp = pd.Series([_ for _ in range(10)]).ewm(span=10, adjust=False).mean()
        for i, x in enumerate(ret):
            assert (x - comp[i]) < 0.001

    def test_ema2(self):
        ret = ts.run(ts.EMA(ts.Func(func4), alpha=1 / 10, adjust=True))
        comp = pd.Series([_ for _ in range(10)]).ewm(alpha=1 / 10, adjust=True).mean()
        for i, x in enumerate(ret):
            assert (x - comp[i]) < 0.001

    def test_last(self):
        assert ts.run(ts.Last(ts.Func(func2))) == [1, 2, 0, 5, 4]

    def test_first(self):
        assert ts.run(ts.First(ts.Func(func2))) == [1, 1, 1, 1, 1]

    def test_last_iter(self):
        assert ts.run(ts.Last(ts.Func(func5))) == [2, 3, 5]

    def test_first_iter(self):
        assert ts.run(ts.First(ts.Func(func5))) == [1, 1, 1]
