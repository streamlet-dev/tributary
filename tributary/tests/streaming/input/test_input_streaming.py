import tributary.streaming as ts


class TestConst:
    def test_const_1(self):
        t = ts.Const(value=1, count=1)
        assert ts.run(t) == [1]

    def test_const_2(self):
        t = ts.Const(value=1, count=5)
        assert ts.run(t) == [1, 1, 1, 1, 1]


class TestTimer:
    def test_timer(self):
        val = 0

        def foo():
            nonlocal val
            val += 1
            return val

        t = ts.Timer(foo, count=5)
        assert ts.run(t) == [1, 2, 3, 4, 5]

        t = ts.Timer(foo, count=5)

    def test_timer_delay(self):
        val = 0

        def foo():
            nonlocal val
            val += 1
            return val

        t = ts.Timer(foo, count=5, interval=0.1)
        assert ts.run(t) == [1, 2, 3, 4, 5]

        t = ts.Timer(foo, count=5)

    def test_timer_generator(self):
        def foo():
            yield 1
            yield 2
            yield 3
            yield 4
            yield 5

        t = ts.Timer(foo)
        assert ts.run(t) == [1]

        t = ts.Timer(foo, count=3)
        assert ts.run(t) == [1, 2, 3]

        t = ts.Timer(foo, count=5)
        assert ts.run(t) == [1, 2, 3, 4, 5]

        t = ts.Timer(foo, count=6)
        assert ts.run(t) == [1, 2, 3, 4, 5]

    def test_timer_generator_delay(self):
        def foo():
            yield 1
            yield 2
            yield 3
            yield 4
            yield 5

        t = ts.Timer(foo, interval=0.1)
        assert ts.run(t) == [1]

        t = ts.Timer(foo, count=3, interval=0.1)
        assert ts.run(t) == [1, 2, 3]

        t = ts.Timer(foo, count=5, interval=0.1)
        assert ts.run(t) == [1, 2, 3, 4, 5]

        t = ts.Timer(foo, count=6, interval=0.1)
        assert ts.run(t) == [1, 2, 3, 4, 5]


class TestFoo:
    def test_timer(self):
        val = 0

        def foo():
            nonlocal val
            val += 1
            return val

        t = ts.Timer(foo, count=5)
        assert ts.run(t) == [1, 2, 3, 4, 5]

        t = ts.Timer(foo, count=5)

    def test_timer_delay(self):
        val = 0

        def foo():
            nonlocal val
            val += 1
            return val

        t = ts.Timer(foo, count=5, interval=0.1)
        assert ts.run(t) == [1, 2, 3, 4, 5]

        t = ts.Timer(foo, count=5)

    def test_foo_generator(self):
        def foo():
            yield 1
            yield 2
            yield 3
            yield 4
            yield 5

        t = ts.Foo(foo)
        assert ts.run(t) == [1, 2, 3, 4, 5]
