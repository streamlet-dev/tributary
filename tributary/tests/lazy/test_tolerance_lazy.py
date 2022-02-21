import tributary.lazy as t


class TestTolerance:
    def test_tolerance(self):
        n = t.Node(value=1.0)
        n2 = n + 1

        assert n() == 1.0
        assert n2() == 2.0

        n.setValue(1.0000000000000001)
        assert n2.isDirty() is False

        n.setValue(1.0001)
        assert n2.isDirty() is True
