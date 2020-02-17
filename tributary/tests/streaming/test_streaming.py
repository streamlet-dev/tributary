import asyncio
import tributary.streaming as ts


class TestStreaming:
    def test_run_simple(self):
        t = ts.Const(value=1, count=1)
        assert ts.run(t) == [1]

    def test_run_foo(self):
        def foo():
            return 5
        t = ts.Foo(foo, count=1)
        assert ts.run(t) == [5]

    def test_run_generator(self):
        def foo():
            yield 1
            yield 2
        t = ts.Foo(foo)
        assert ts.run(t) == [1, 2]

    def test_run_async_foo(self):
        async def foo():
            await asyncio.sleep(.1)
            return 5

        t = ts.Foo(foo, count=1)
        assert ts.run(t) == [5]

    def test_run_async_generator(self):
        async def foo():
            yield 1
            yield 2
        t = ts.Foo(foo)
        assert ts.run(t) == [1, 2]
