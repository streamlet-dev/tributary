import tributary.lazy as tl
from datetime import datetime
from time import sleep


def func():
    yield 1
    yield 2


class TestUtils:
    def test_expire(self):
        n = tl.Node(value=5)
        n2 = n + 1
        sec = datetime.now().second
        out = tl.Expire(n2, second=(sec + 2) % 60)

        # assert initial value
        assert out() == 6

        # set new value
        n.setValue(6)

        # continue to use old value until 2+ seconds elapsed
        assert out() == 6

        sleep(3)
        assert out() == 7

    def test_interval(self):
        n = tl.Node(value=5)
        n2 = n + 1
        out = tl.Interval(n2, seconds=2)

        # # assert initial value
        assert out() == 6

        # # set new value
        n.setValue(6)

        # continue to use old value until 2+ seconds elapsed
        assert out() == 6

        sleep(3)
        assert out() == 7

    def test_window_any_size(self):
        n = tl.Window(tl.Node(value=func))

        assert n() == [1]
        assert n() == [1, 2]

    def test_window_fixed_size(self):
        n = tl.Window(tl.Node(value=func), size=2)
        assert n() == [1]
        assert n() == [1, 2]

    def test_window_fixed_size_full_only(self):
        n = tl.Window(tl.Node(value=func), size=2, full_only=True)
        assert n() is None
        assert n() == [1, 2]
