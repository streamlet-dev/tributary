import tributary.streaming as ts


def foo():
    yield 1
    yield 1
    yield 1
    yield 1
    yield 1


class TestRolling:
    def test_count(self):
        assert ts.run(ts.Count(ts.Foo(foo))) == [1, 2, 3, 4, 5]

    def test_sum(self):
        assert ts.run(ts.Sum(ts.Foo(foo))) == [1, 2, 3, 4, 5]
