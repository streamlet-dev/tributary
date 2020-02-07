import tributary.lazy as t
import random


class Foo1(t.BaseClass):
    def __init__(self, *args, **kwargs):
        self.x = self.node('x', readonly=False, default_or_starting_value=1, trace=True)


class Foo2(t.BaseClass):
    def __init__(self, *args, **kwargs):
        self.y = self.node('y', readonly=False, default_or_starting_value=2, trace=True)

        # ensure no __nodes clobber
        self.test = self.node('test', readonly=False, default_or_starting_value=2, trace=True)
        self.x = self.node('x', readonly=False, default_or_starting_value=2, trace=True)


class Foo3(t.BaseClass):
    @t.node(trace=True)
    def foo1(self):
        return self.random()  # test self access

    def random(self):
        return random.random()

    @t.node(trace=True)
    def foo3(self, x=4):
        return 3 + x


class Foo4(t.BaseClass):
    @t.node(trace=True)
    def foo1(self):
        return self.foo2() + 1

    @t.node(trace=True)
    def foo2(self):
        return random.random()


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
        assert z._dirty or z._subtree_dirty()
        print(z())  # should call foo3 and recompute (first time)
        assert not (z._dirty or z._subtree_dirty())
        print(z())  # should not recompute (foo3 unchanged)
        f3.set(x=0)
        assert z._dirty or z._subtree_dirty()
        print(z())  # should call foo3 and recompute (value changed)
        assert not (z._dirty or z._subtree_dirty())
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
        assert z.networkx()
