import tributary.asynchronous as t


class TestOps:
    def test_unary(self):
        def foo():
            return True

        unary = t.unary(lambda x: not x, foo)
        out = t.run(unary)
        assert len(out) == 1
        assert out[-1] == False

    def test_bin(self):
        def foo1():
            return 1

        def foo2():
            return 1

        bin = t.bin(lambda x, y: x + y, foo1, foo2)
        out = t.run(bin)
        assert len(out) == 1
        assert out[-1] == 2

    def test_noop(self):
        def foo():
            return 5

        noop = t.Noop(foo)
        out = t.run(noop)
        assert len(out) == 1
        assert out[-1] == 5

    def test_negate(self):
        def foo():
            return 5

        neg = t.Negate(foo)
        out = t.run(neg)
        assert len(out) == 1
        assert out[-1] == -5

    def test_invert(self):
        def foo():
            return 5

        inv = t.Invert(foo)
        out = t.run(inv)
        assert len(out) == 1
        assert out[-1] == 1/5

    def test_not(self):
        def foo():
            return True

        inv = t.Not(foo)
        out = t.run(inv)
        assert len(out) == 1
        assert out[-1] == False

    def test_add(self):
        def foo1():
            return 1

        def foo2():
            return 1

        add = t.Add(foo1, foo2)
        out = t.run(add)
        assert len(out) == 1
        assert out[-1] == 2

    def test_sub(self):
        def foo1():
            return 1

        def foo2():
            return 1

        sub = t.Sub(foo1, foo2)
        out = t.run(sub)
        assert len(out) == 1
        assert out[-1] == 0

    def test_mult(self):
        def foo1():
            return 2

        def foo2():
            return 3

        mult = t.Mult(foo1, foo2)
        out = t.run(mult)
        assert len(out) == 1
        assert out[-1] == 6

    def test_div(self):
        def foo1():
            return 3

        def foo2():
            return 2

        div = t.Div(foo1, foo2)
        out = t.run(div)
        assert len(out) == 1
        assert out[-1] == 1.5

    def test_mod(self):
        def foo1():
            return 3

        def foo2():
            return 2

        mod = t.Mod(foo1, foo2)
        out = t.run(mod)
        assert len(out) == 1
        assert out[-1] == 1

    def test_pow(self):
        def foo1():
            return 3

        def foo2():
            return 2

        pow = t.Pow(foo1, foo2)
        out = t.run(pow)
        assert len(out) == 1
        assert out[-1] == 9

    def test_and(self):
        def foo1():
            return True

        def foo2():
            return False

        out = t.run(t.And(foo1, foo2))
        assert len(out) == 1
        assert out[-1] == False

    def test_or(self):
        def foo1():
            return True

        def foo2():
            return False

        out = t.run(t.Or(foo1, foo2))
        assert len(out) == 1
        assert out[-1] == True

    def test_equal(self):
        def foo1():
            return True

        def foo2():
            return True

        out = t.run(t.Equal(foo1, foo2))
        assert len(out) == 1
        assert out[-1] == True

    def test_less(self):
        def foo1():
            return 2

        def foo2():
            return 3

        out = t.run(t.Less(foo1, foo2))
        assert len(out) == 1
        assert out[-1] == True

    def test_more(self):
        def foo1():
            return 2

        def foo2():
            return 3

        out = t.run(t.More(foo1, foo2))
        assert len(out) == 1
        assert out[-1] == False
