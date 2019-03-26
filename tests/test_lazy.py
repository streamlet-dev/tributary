import tributary.lazy as t
import sys

if sys.version_info[0] > 2:
    class Foo1(t.BaseClass):
        def __init__(self, *args, **kwargs):
            self.x = self.node('x', readonly=False, default_or_starting_value=1, trace=True)

    class Foo2(t.BaseClass):
        def __init__(self, *args, **kwargs):
            self.y = self.node('y', readonly=False, default_or_starting_value=2, trace=True)

    class TestConfig:
        def setup(self):
            pass
            # setup() before each test method

        def test_1(self):
            f1 = Foo1()
            f2 = Foo2()
            z = f1.x + f2.y
            assert z.value() == 3
            f1.x = 2
            assert z.value() == 4
            f2.y = 4
            assert z.value() == 6

        def test_2(self):
            f1 = Foo1()
            f2 = Foo1()
            f3 = Foo1()
            z = f1.x + f2.x - f3.x
            assert z.value() == 1
            f1.x = 2
            assert z.value() == 2
            f2.x = 4
            f3.x = 4
            assert z.value() == 2
