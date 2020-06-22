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

    def test_deep_bfs(self):
        a = ts.Const(1, count=1)
        b = ts.Random()
        c = ts.Curve([1, 2, 3])

        d = a + b
        e = a + c
        f = b + c

        g = ts.Print(d)
        h = ts.Print(e)
        i = ts.Print(f)

        def _ids(lst):
            return set([l._id for l in lst])

        def _ids_ids(lst_of_list):
            ret = []
            for lst in lst_of_list:
                ret.append(_ids(lst))
            return ret

        assert _ids(a._deep_bfs(tops_only=True)) == _ids([a, b, c])
        assert _ids(b._deep_bfs(tops_only=True)) == _ids([a, b, c])
        assert _ids(c._deep_bfs(tops_only=True)) == _ids([a, b, c])
        assert _ids(d._deep_bfs(tops_only=True)) == _ids([a, b, c])
        assert _ids(e._deep_bfs(tops_only=True)) == _ids([a, b, c])
        assert _ids(f._deep_bfs(tops_only=True)) == _ids([a, b, c])
        assert _ids(g._deep_bfs(tops_only=True)) == _ids([a, b, c])
        assert _ids(h._deep_bfs(tops_only=True)) == _ids([a, b, c])
        assert _ids(i._deep_bfs(tops_only=True)) == _ids([a, b, c])

        for x in (a, b, c, d, e, f, g, h, i):
            for y in (a, b, c, d, e, f, g, h, i):
                assert _ids_ids(x._deep_bfs()) == _ids_ids(y._deep_bfs())
