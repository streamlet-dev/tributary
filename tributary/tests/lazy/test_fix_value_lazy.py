import random

import tributary.lazy as t


class TestCallableLock:
    def test_callable_lock(self):
        n = t.Node(value=random.random, dynamic=True)

        x = n()
        assert n() != x

        n.setValue(5)
        assert n() == 5

        n.unlock()
        assert n() != 5
