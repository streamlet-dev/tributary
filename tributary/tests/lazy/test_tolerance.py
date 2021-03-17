import tributary.lazy as t


class TestTolerance:
    def test_tolerance(self):
        n = t.Node(value=1.0)
        assert n() == 1.0

        n.setValue(1.0000000000000001)
        assert n.isDirty() is False

        n.setValue(1.0001)
        assert n.isDirty() is True
