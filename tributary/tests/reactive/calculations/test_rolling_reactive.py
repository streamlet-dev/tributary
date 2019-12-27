import tributary as t


class TestRolling:
    def test_count(self):
        def foo():
            yield 1
            yield 2
            yield 3

        out = t.run(t.Count(foo))
        assert len(out) == 3
        assert out[-1] == 3
        assert out[-2] == 2
        assert out[-3] == 1

    def test_sub(self):
        def foo():
            yield 1
            yield 2
            yield 3

        out = t.run(t.Sum(foo))
        assert len(out) == 3
        assert out[-1] == 6
        assert out[-2] == 3
        assert out[-3] == 1
