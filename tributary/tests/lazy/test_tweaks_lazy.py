import tributary.lazy as tl


class TestLazyTweaks:
    def test_tweaks(self):
        n = tl.Node(name="Test", value=5)
        n2 = n + 1

        # base operation
        print(n2())
        assert n2() == 6

        print(n2(1))
        assert n2.isDirty()

        # tweaking operation applied to `n`
        assert n2(1) == 2

        # not permanently set
        assert n2() == 6

    def test_tweaks_dirtiness_and_parent(self):
        n1 = tl.Node(value=1, name="n1")
        n2 = n1 + 2
        n3 = n2 + 4

        assert n3({n1: -1}) == 5
        assert n3() == 7
        assert n3({n2: 2}) == 6
        assert n3(2) == 6
        assert n3(2, 2) == 4
        assert n3({n1: -1}) == 5
        assert n3({n1: -1}) == 5
        assert n3() == 7
        assert n3({n2: 2}) == 6
        assert n3(2) == 6
        assert n3(2, 2) == 4
        assert n3({n1: -1}) == 5
        assert n3() == 7
        assert n3() == 7
        assert n3() == 7
