import tributary.lazy as tl


def foo4():
    yield [1]
    yield [1, 2]
    yield [1, 2, 3]


class TestBasket:
    def test_Len(self):
        t = tl.Node(callable=foo4)
        out = tl.Len(t)
        assert out() == 1
        assert out() == 2
        assert out() == 3

    def test_Count(self):
        t = tl.Node(callable=foo4)
        out = tl.CountBasket(t)
        assert out() == 1
        assert out() == 2
        assert out() == 3

    def test_Max(self):
        t = tl.Node(callable=foo4)
        out = tl.MaxBasket(t)
        assert out() == 1
        assert out() == 2
        assert out() == 3

    def test_Min(self):
        t = tl.Node(callable=foo4)
        out = tl.MinBasket(t)
        assert out() == 1
        assert out() == 1
        assert out() == 1

    def test_Average(self):
        t = tl.Node(callable=foo4)
        out = tl.AverageBasket(t)
        assert out() == 1
        assert out() == 1.5
        assert out() == 2
