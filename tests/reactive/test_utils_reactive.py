import tributary as t
from datetime import datetime, timedelta


class TestUtils:
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

    def test_delay(self):
        def foo():
            return 1

        delay = t.Delay(foo, delay=1)
        first = datetime.now()
        out = t.run(delay)
        last = datetime.now()
        assert last - first < timedelta(seconds=2)
        assert last - first > timedelta(seconds=1)
        assert len(out) == 1

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

    def test_unroll(self):
        def ran():
            return [1, 2, 3]

        w = t.Unroll(ran)
        out = t.run(w)
        assert len(out) == 3
        assert out[-1] == 3

    def test_unrollDF(self):
        import pandas as pd
        df = pd.DataFrame(pd.util.testing.makeTimeSeries())

        def ran():
            return df

        w = t.UnrollDataFrame(ran)
        out = t.run(w)
        assert len(out) == 30

    def test_merge1(self):
        def foo1():
            return 1

        def foo2():
            return 2

        m = t.Merge(foo1, foo2)
        out = t.run(m)
        assert len(out) == 1
        assert out[-1] == [1, 2]

    def test_merge2(self):
        def foo1():
            yield 1

        def foo2():
            return 2

        m = t.Merge(foo1, foo2)
        out = t.run(m)
        assert len(out) == 1
        assert out[-1] == [1, 2]

    def test_merge3(self):
        def foo1():
            return 1

        def foo2():
            yield 2

        m = t.Merge(foo1, foo2)
        out = t.run(m)
        assert len(out) == 1
        assert out[-1] == [1, 2]

    def test_merge4(self):
        def foo1():
            yield 1

        def foo2():
            yield 2

        m = t.Merge(foo1, foo2)
        out = t.run(m)
        assert len(out) == 1
        assert out[-1] == [1, 2]

    def test_list_merge1(self):
        def foo1():
            return [1]

        def foo2():
            return [2]

        m = t.ListMerge(foo1, foo2)
        out = t.run(m)
        assert len(out) == 1
        assert out[-1] == [1, 2]

    def test_list_merge2(self):
        def foo1():
            yield [1]

        def foo2():
            return [2]

        m = t.ListMerge(foo1, foo2)
        out = t.run(m)
        assert len(out) == 1
        assert out[-1] == [1, 2]

    def test_list_merge3(self):
        def foo1():
            return [1]

        def foo2():
            yield [2]

        m = t.ListMerge(foo1, foo2)
        out = t.run(m)
        assert len(out) == 1
        assert out[-1] == [1, 2]

    def test_list_merge4(self):
        def foo1():
            yield [1]

        def foo2():
            yield [2]

        m = t.ListMerge(foo1, foo2)
        out = t.run(m)
        assert len(out) == 1
        assert out[-1] == [1, 2]

    def test_dict_merge1(self):
        def foo1():
            return {'a': 1}

        def foo2():
            return {'b': 2}

        m = t.DictMerge(foo1, foo2)
        out = t.run(m)
        assert len(out) == 1
        assert out[-1] == {'a': 1, 'b': 2}

    def test_dict_merge2(self):
        def foo1():
            yield {'a': 1}

        def foo2():
            return {'b': 2}

        m = t.DictMerge(foo1, foo2)
        out = t.run(m)
        assert len(out) == 1
        assert out[-1] == {'a': 1, 'b': 2}

    def test_dict_merge3(self):
        def foo1():
            return {'a': 1}

        def foo2():
            yield {'b': 2}

        m = t.DictMerge(foo1, foo2)
        out = t.run(m)
        assert len(out) == 1
        assert out[-1] == {'a': 1, 'b': 2}

    def test_dict_merge4(self):
        def foo1():
            yield {'a': 1}

        def foo2():
            yield {'b': 2}

        m = t.DictMerge(foo1, foo2)
        out = t.run(m)
        assert len(out) == 1
        assert out[-1] == {'a': 1, 'b': 2}

    def test_reduce(self):
        def foo1():
            return {'b': 2}

        def foo2():
            yield {'c': 3}

        out = t.run(t.Reduce({'a': 1}, foo1, foo2))
        assert len(out) == 1
        assert out[-1] == [{'a': 1}, {'b': 2}, {'c': 3}]

    def test_ensure_count(self):
        import tributary.reactive as tr

        def myfoo(state, data):
            state.count = state.count + 1
            return data + state.count

        def myfoo2(data):
            return data + 1

        out1 = tr.run(tr.Const(1))
        out2 = tr.run(tr.Apply(myfoo2, tr.Const(1)))
        out3 = tr.run(tr.Apply(tr.State(myfoo, count=0), tr.Const(1)))
        print(out1)
        print(out2)
        print(out3)
        assert len(out1) == 1
        assert len(out2) == 1
        assert len(out3) == 1
