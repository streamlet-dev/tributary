import tributary.lazy as tl
from datetime import datetime
from time import sleep


class TestUtils:
    def test_expire(self):
        n = tl.Node(value=5)
        sec = datetime.now().second
        out = tl.Expire(n, second=(sec + 2) % 60)

        # assert initial value
        assert out() == 5

        # set new value
        n.setValue(6)

        # continue to use old value until 2+ seconds elapsed
        assert out() == 5

        sleep(3)
        assert out() == 6

    def test_interval(self):
        n = tl.Node(value=5)
        out = tl.Interval(n, seconds=2)

        # assert initial value
        assert out() == 5

        # set new value
        n.setValue(6)

        # continue to use old value until 2+ seconds elapsed
        assert out() == 5

        sleep(3)
        assert out() == 6
