import asyncio
import json as JSON
import os.path
import pytest
import sys
import time
import tributary.streaming as ts
from asyncio import sleep


def foo():
    yield 1
    yield 2


def foo2():
    yield [1, 2]
    yield [3, 4]


class TestUtils:
    def setup(self):
        time.sleep(0.1)

    def test_delay(self):
        out = ts.Delay(ts.Foo(foo), delay=5)
        now = time.time()
        ret = ts.run(out)
        assert time.time() - now > 5
        assert ret == [1, 2]

    def test_apply(self):
        def square(val):
            return val ** 2

        assert ts.run(ts.Apply(ts.Foo(foo), foo=square)) == [1, 4]

    def test_window_any_size(self):
        assert ts.run(ts.Window(ts.Foo(foo))) == [[1], [1, 2]]

    def test_window_fixed_size(self):
        assert ts.run(ts.Window(ts.Foo(foo), size=2)) == [[1], [1, 2]]

    def test_window_fixed_size_full_only(self):
        assert ts.run(ts.Window(ts.Foo(foo), size=2, full_only=True)) == [[1, 2]]

    def test_unroll(self):
        assert ts.run(ts.Unroll(ts.Foo(foo2))) == [1, 2, 3, 4]

    def test_merge(self):
        def foo1():
            yield 1
            yield 3

        def foo2():
            yield 2
            yield 4
            yield 6

        out = ts.Merge(ts.Print(ts.Foo(foo1)), ts.Print(ts.Foo(foo2)))
        assert ts.run(out) == [(1, 2), (3, 4)]

    def test_fixed_map(self):
        def foo():
            yield [1, 2, 3]
            yield [4, 5, 6]

        out = ts.Reduce(*[x + 1 for x in ts.Foo(foo).map(3)])
        assert ts.run(out) == [(2, 3, 4), (5, 6, 7)]

    def test_reduce(self):
        def foo1():
            yield 1
            yield 4

        def foo2():
            yield 2
            yield 5
            yield 7

        def foo3():
            yield 3
            yield 6
            yield 8

        out = ts.Reduce(ts.Foo(foo1), ts.Foo(foo2), ts.Foo(foo3))
        assert ts.run(out) == [(1, 2, 3), (4, 5, 6)]

    def test_reduce_stateful(self):
        def foo1():
            yield 0
            yield 2
            yield 4

        def foo2():
            yield 1
            yield 3
            yield 5

        def foo3():
            yield 2
            yield 4
            yield 6

        def reduce(node1_value, node2_value, node3_value, reducer_node):

            if not hasattr(reducer_node, "state"):
                # on first call, make sure node tracks state
                reducer_node.set("state", {"n1": None, "n2": None, "n3": None})

            if node1_value is not None:
                reducer_node.state["n1"] = node1_value

            if node2_value is not None:
                reducer_node.state["n2"] = node2_value

            if node3_value is not None:
                reducer_node.state["n3"] = node3_value

            return reducer_node.state

        n1 = ts.Foo(foo1)
        n2 = ts.Foo(foo2)
        n3 = ts.Foo(foo3)

        r = ts.Reduce(n1, n2, n3, reducer=reduce, inject_node=True)
        out = ts.run(r)

        print(out)
        assert out == [
            {"n1": 0, "n2": 1, "n3": 2},
            {"n1": 2, "n2": 3, "n3": 4},
            {"n1": 4, "n2": 5, "n3": 6},
        ]

    @pytest.mark.skipif(
        (sys.version_info.major, sys.version_info.minor) == (3, 7) and os.name == "nt",
        reason="bug on windows py3.7 where pytest hangs",
    )
    def test_process(self):
        def foo():
            yield {"a": 1, "b": 2}
            yield {"a": 2, "b": 4}
            yield {"a": 3, "b": 6}
            yield {"a": 4, "b": 8}

        def _json(val):
            return JSON.dumps(val)

        cmd = "{} {} --1".format(
            sys.executable, os.path.join(os.path.dirname(__file__), "echo.py")
        )
        print(cmd)

        ret = ts.run(
            ts.Subprocess(ts.Foo(foo).print("in:"), cmd, json=True).print("out:")
        )

        print(ret)
        assert ret == [
            {"a": 1, "b": 2},
            {"a": 2, "b": 4},
            {"a": 3, "b": 6},
            {"a": 4, "b": 8},
        ]

    def test_interval(self):
        def reducer(short, long, node):
            if not node.has("state"):
                node.set("state", {"long": 0, "short": 0})
            if long:
                node.state["long"] += long
            if short:
                node.state["short"] += short
            return node.state.copy()

        async def short():
            await sleep(1)
            return 1

        async def long():
            await sleep(2)
            return 2

        def interval(foo, time=1):
            task = None

            async def _ret():
                nonlocal task
                if task is None:
                    task = asyncio.ensure_future(foo())

                if not task.done():
                    await sleep(time)
                    return None

                result = task.result()
                task = None

                return result

            return _ret

        short_node = ts.Foo(interval(short, 1), count=5)
        long_node = ts.Foo(interval(long, 1), count=5)
        out = ts.Reduce(
            short_node, long_node, reducer=reducer, inject_node=True
        ).print()
        ts.run(out)

    def test_debounce(self):
        async def clic():
            for _ in range(10):
                await sleep(0.1)
                yield 1

        n = ts.Node(clic).debounce()
        out = ts.run(n)
        assert out == [1]

    def test_throttle(self):
        async def clic():
            for _ in range(10):
                await sleep(1)
                yield 1

        n = ts.Node(clic).throttle(2)
        out = ts.run(n)
        assert out == [[1], [1, 1], [1, 1], [1, 1], [1, 1]]
