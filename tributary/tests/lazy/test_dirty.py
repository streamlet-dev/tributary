import tributary.lazy as t


class Foo5(t.LazyGraph):
    @t.node()
    def z(self):
        return self.x | self.y()

    @t.node()
    def y(self):
        return 10

    def reset(self):
        self.x = None

    def __init__(self):
        self.x = self.node(name="x", value=None)



class TestDirtyPropogation:
    def test_or_dirtypropogation(self):
        f = Foo5()
        assert f.z()() == 10
        assert f.x() is None

        f.x = 5

        assert f.x() == 5
        assert f.z()() == 5

        f.reset()

        assert f.x() is None
        assert f.z()() == 10
