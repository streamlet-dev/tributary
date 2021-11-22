import tributary.lazy as t


class TestDeclarative:
    def test_simple_declarative(self):
        n = t.Node(value=1)
        z = n + 5
        assert z() == 6
        n.setValue(2)
        assert z() == 7
