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


class TestLazy:
    def test_misc(self):
        f4 = Foo4()
        z = f4.foo1()
        assert z.print()
        assert z.graph()
        assert z.graphviz()
