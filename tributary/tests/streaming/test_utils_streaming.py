import asyncio
import json as JSON
import os.path
import pytest
import sys
import time
import tributary.streaming as ts


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
