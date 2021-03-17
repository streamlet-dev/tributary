import tributary.lazy as tl


class TestLazyTweaks:
    def test_tweaks(self):
        n = tl.Node(name="Test", value=5)
        n2 = n + 1

        # base operation
        print(n2())
        assert n2() == 6

        print(n2(1))
        assert n2.isDirty({n: 1})

        # tweaking operation applied to `n`
        assert n2(1) == 2

        # not permanently set
        assert n2() == 6
