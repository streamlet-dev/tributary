import tributary.lazy as t
import random


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
    @t.node(trace=True)
    def foo1(self):
        return self.random()  # test self access

    def random(self):
        return random.random()

    @t.node(trace=True)
    def foo3(self, x=4):
        return 3 + x


class Foo4(t.LazyGraph):
    @t.node(trace=True)
    def foo1(self):
        return self.foo2() + 1

    @t.node(trace=True)
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
    def test_function(self):
        f1 = Foo1()
        f3 = Foo3()
        z = f1.x + f3.foo1()
        assert z._dirty or z._subtree_dirty()
        print(z())  # should call foo1 and recompute
        assert not (z._dirty or z._subtree_dirty())
        print(z())  # should call foo1 and recompute again
        f1.x = 10
        assert z._dirty or z._subtree_dirty()
        print(z())  # should call foo1 and recompute again
        assert not (z._dirty or z._subtree_dirty())
        print(z())  # should call foo1 and recompute again

    def test_function2(self):
        f1 = Foo1()
        f3 = Foo3()
        z = f1.x + f3.foo3()
        assert z._dirty or z._subtree_dirty()
        print(z())  # should call foo3 and recompute (first time)
        assert not (z._dirty or z._subtree_dirty())
        print(z())  # should not recompute (foo3 unchanged)
        f1.x = 10
        assert z._dirty or z._subtree_dirty()
        print(z())  # should call foo3 and recompute
        assert not (z._dirty or z._subtree_dirty())
        print(z())  # should not recompute (x unchanged)

    def test_method(self):
        f1 = Foo1()
        f3 = Foo3().foo3()
        z = f1.x + f3
        assert z.isDirty()
        print(z())  # should call foo3 and recompute (first time)
        assert not z.isDirty()
        print(z())  # should not recompute (foo3 unchanged)
        f3.set(x=0)
        assert z.isDirty()
        print(z())  # should call foo3 and recompute (value changed)
        assert not z.isDirty()
        print(z())  # should not recompute (value unchanged)

    def test_method2(self):
        f4 = Foo4()
        z = f4.foo1()
        assert z._dirty or z._subtree_dirty()
        print(z())
        assert not (z._dirty or z._subtree_dirty())
        print(z())

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
