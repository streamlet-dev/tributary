def conditionals():
    yield True
    yield False
    yield True


def if_stream():
    yield 1
    yield -1
    yield 3


def else_stream():
    yield -1
    yield 2
    yield -1


class TestConditional:
    def test_if(self):
        import tributary.streaming as ts

        assert ts.run(
            ts.Print(
                ts.If(ts.Foo(conditionals), ts.Foo(if_stream), ts.Foo(else_stream))
            )
        ) == [1, 2, 3]
