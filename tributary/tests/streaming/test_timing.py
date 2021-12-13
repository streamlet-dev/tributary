import tributary.streaming as ts


def func():
    yield 1
    yield 2
    yield 3
    yield 4
    yield 5
    yield 6


def func2():
    yield 1
    yield ts.StreamNone()
    yield 10
    yield ts.StreamNone()
    yield 100
    yield 1000


class TestTiming:
    def test_normal(self):
        out = ts.Func(func) + ts.Func(func2)
        assert ts.run(out) == [2, 12, 103, 1004]

    def test_drop(self):
        out = ts.Func(func, drop=True) + ts.Func(func2)
        assert ts.run(out) == [2, 12, 104, 1006]

    def test_replace(self):
        out = ts.Func(func, replace=True) + ts.Func(func2)
        assert ts.run(out) == [2, 13, 105, 1006]

    def test_repeat(self):
        out = ts.Func(func) + ts.Func(func2, repeat=True)
        assert ts.run(out) == [2, 3, 13, 14, 105, 1006]
