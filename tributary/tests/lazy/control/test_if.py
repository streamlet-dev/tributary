def foo():
    yield True
    yield False
    yield True


class TestConditional:
    def test_if(self):
        import tributary.lazy as tl

        x = tl.Node(callable=foo)
        y = tl.Node(1)
        z = tl.Node(-1)

        out = tl.If(x, y, z)

        print(out.graph())
        assert out() == 1
        assert out() == -1
        assert out() == 1
        assert out() == 1
        assert out() == 1
