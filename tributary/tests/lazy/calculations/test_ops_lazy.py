import tributary as t


class Foo1(t.LazyGraph):
    def __init__(self, *args, **kwargs):
        self.x = self.node('x', readonly=False, value=1, trace=True)


class Foo2(t.LazyGraph):
    def __init__(self, *args, **kwargs):
        self.y = self.node('y', readonly=False, value=2, trace=True)

        # ensure no __nodes clobber
        self.test = self.node('test', readonly=False, value=2, trace=True)
        self.x = self.node('x', readonly=False, value=2, trace=True)


class Foo3(t.LazyGraph):
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


class TestLazyOps:
    def test_add(self):
        f1 = Foo1()
        f2 = Foo2()
        z = f1.x + f2.y
        assert z() == 3
        f1.x = 2
        assert z() == 4
        f2.y = 4
        assert z() == 6

    def test_multi(self):
        f1 = Foo1()
        f2 = Foo1()
        f3 = Foo1()
        z = f1.x + f2.x - f3.x
        assert z() == 1
        f1.x = 2
        assert z() == 2
        f2.x = 4
        f3.x = 4
        assert z() == 2

    def test_or(self):
        f = Foo3()
        assert f.z()() == 10
        assert f.x() is None

        f.x = 5

        assert f.x() == 5
        assert f.z()() == 5

        f.reset()

        assert f.x() is None
        assert f.z()() == 10
