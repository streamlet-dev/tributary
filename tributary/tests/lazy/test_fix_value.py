import tributary.lazy as t
import random


class TestCallableLock:
    def test_callable_lock(self):
        n = t.Node(callable=lambda: random.random())

        x = n()
        assert n() != x

        n.setValue(5)
        assert n() == 5

        n.unlock()
        assert n() != 5
