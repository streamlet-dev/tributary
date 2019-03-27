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


class TestConfig:
    def setup(self):
        pass
        # setup() before each test method

    def test_1(self):
        f1 = Foo1()
        f2 = Foo2()
        z = f1.x + f2.y
        assert z() == 3
        f1.x = 2
        assert z() == 4
        f2.y = 4
        assert z() == 6

    def test_2(self):
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

    def test_3(self):
        f1 = Foo1()
        f3 = Foo3()
        z = f1.x + f3.foo1()
        print(z())  # should call foo1 and recompute
        print(z())  # should call foo1 and recompute again
        f1.x = 10
        print(z())  # should call foo1 and recompute again
        print(z())  # should call foo1 and recompute again

    def test_4(self):
        f1 = Foo1()
        f3 = Foo3()
        z = f1.x + f3.foo3()
        print(z())  # should call foo3 and recompute (first time)
        print(z())  # should not recompute (foo3 unchanged)
        f1.x = 10
        print(z())  # should call foo3 and recompute
        print(z())  # should not recompute (x unchanged)

    def test_5(self):
        f1 = Foo1()
        f3 = Foo3().foo3()
        z = f1.x + f3
        print(z())  # should call foo3 and recompute (first time)
        print(z())  # should not recompute (foo3 unchanged)
        f3.set(x=0)
        print(z())  # should call foo3 and recompute (value changed)
        print(z())  # should not recompute (value unchanged)

    def test_6(self):
        f4 = Foo4()
        z = f4.foo1()
        print('recompute?')
        print(z())
        print('recompute?')
        print(z())
