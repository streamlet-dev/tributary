import tributary.asynchronous as t
from datetime import datetime, timedelta


class TestConfig:
    def test_timer(self):
        def foo():
            return 1

        timer = t.Timer(foo, {}, 1, 2)
        first = datetime.now()
        out = t.run(timer)
        last = datetime.now()
        print(last-first)
        assert last - first < timedelta(seconds=3)
        assert last - first > timedelta(seconds=2)
        assert len(out) == 2

    def test_state(self):
        def stream(state):
            for i in range(10):
                yield i + state.val

        f = t.Foo(t.State(stream, val=5))
        out = t.run(f)
        assert len(out) == 10
        assert out[-1] == 14
        assert out[-2] == 13
        assert out[0] == 5

    def test_apply(self):
        def myfoo(state, data):
            state.count = state.count + 1
            return data + state.count

        f = t.Apply(t.State(myfoo, count=0), t.Const(1))
        out = t.run(f)
        assert len(out) == 1

    def test_window(self):
        def ran():
            for i in range(10):
                yield i

        w = t.Window(ran, size=3, full_only=True)
        out = t.run(w)
        assert len(out) == 8
        assert out[-1] == [7, 8, 9]
