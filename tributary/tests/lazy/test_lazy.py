import tributary.lazy as t
import random


class Foo1(t.LazyGraph):
    def __init__(self, *args, **kwargs):
        self.x = self.node("x", readonly=False, value=1)


class Foo2(t.LazyGraph):
    def __init__(self, *args, **kwargs):
        self.y = self.node("y", readonly=False, value=2)

        # ensure no __nodes clobber
        self.test = self.node("test", readonly=False, value=2)
        self.x = self.node("x", readonly=False, value=2)


class Foo3(t.LazyGraph):
    @t.node()
    def foo1(self):
        return self.random()  # test self access

    def random(self):
        return random.random()

    @t.node()
    def foo3(self, x=4):
        return 3 + x


class Foo4(t.LazyGraph):
    @t.node()
    def foo1(self):
        return self.foo2() + 1

    @t.node()
    def foo2(self):
        return random.random()


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


class TestLazy:
    def test_misc(self):
        f4 = Foo4()
        z = f4.foo1()
        assert z.print()
        assert z.graph()
        assert z.graphviz()


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


class TestDeclarative:
    def test_simple_declarative(self):
        n = t.Node(value=1)
        z = n + 5
        assert z() == 6
        n.setValue(2)
        assert z() == 7
