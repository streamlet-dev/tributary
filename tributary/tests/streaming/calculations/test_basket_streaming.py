import tributary.streaming as ts


def foo4():
    yield [1]
    yield [1, 2]
    yield [1, 2, 3]


class TestBasket:
    def test_Len(self):
        t = ts.Timer(foo4, count=3)
        out = ts.Len(t)
        assert ts.run(out) == [1, 2, 3]

    def test_Count(self):
        t = ts.Timer(foo4, count=3)
        out = ts.CountBasket(t)
        assert ts.run(out) == [1, 2, 3]

    def test_Max(self):
        t = ts.Timer(foo4, count=3)
        out = ts.MaxBasket(t)
        assert ts.run(out) == [1, 2, 3]

    def test_Min(self):
        t = ts.Timer(foo4, count=3)
        out = ts.MinBasket(t)
        assert ts.run(out) == [1, 1, 1]

    def test_Average(self):
        t = ts.Timer(foo4, count=3)
        out = ts.AverageBasket(t)
        assert ts.run(out) == [1, 1.5, 2]
