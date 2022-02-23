import tributary.lazy as t
import random


class Func1(t.LazyGraph):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.x = self.node("x", readonly=False, value=1)


class Func2(t.LazyGraph):
    def __init__(self, *args, **kwargs):
        self.y = self.node("y", readonly=False, value=2)

        # ensure no __nodes clobber
        self.test = self.node("test", readonly=False, value=2)
        self.x = self.node("x", readonly=False, value=2)


class Func3(t.LazyGraph):
    @t.node()
    def func1(self):
        return self.random()  # test self access

    def random(self):
        return random.random()

    @t.node()
    def func3(self, x=4):
        return 3 + x


class Func4(t.LazyGraph):
    @t.node()
    def func1(self):
        return self.func2() + 1

    @t.node()
    def func2(self):
        return random.random()


class Func5(t.LazyGraph):
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
        f4 = Func4()
        z = f4.func1()
        assert isinstance(z, float) and z >= 1
        assert f4.func1.print()
        assert f4.func1.graph()
        assert f4.func1.graphviz()

    def test_lazy_default_func_arg(self):
        def func(val, prev_val=0):
            print("val:\t{}".format(val))
            print("prev_val:\t{}".format(prev_val))
            return val + prev_val

        n = t.Node(value=func)
        n.kwargs["val"].setValue(5)

        assert n() == 5

        n.set(prev_val=100)

        assert n() == 105

    def test_lazy_args_by_name_and_arg(self):
        # see the extended note in lazy.node about callable_args_mapping
        n = t.Node(name="Test", value=5)
        n2 = n + 1

        assert n2.kwargs["x"]._name_no_id == "Test"

    def test_or_dirtypropogation(self):
        f = Func5()
        assert f.z()() == 10
        assert f.x() is None

        f.x = 5

        assert f.x() == 5
        assert f.z()() == 5

        f.reset()

        assert f.x() is None
        assert f.z()() == 10
