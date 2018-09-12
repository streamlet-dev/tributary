import tributary as t
import random
import time


class TestConfig:
    def setup(self):
        pass
        # setup() before each test method

    def test_1(self):
        def foo():
            return random.random()

        print('''
        ******************************
        *           test             *
        ******************************
        ''')
        test = t.Timer(foo, {}, 0, 5)
        test2 = t.Negate(t.Share(test))
        res2 = t.Add(test, test2)
        p2 = t.Print(res2)
        t.run(p2)

    def test_2(self):
        print('''
        ******************************
        *           test2            *
        ******************************
        ''')

        def foo():
            return random.random()

        def long():
            print('long called')
            time.sleep(.1)
            return 5

        rand = t.Timer(foo, interval=0, repeat=5)
        five = t.Timer(long, interval=0, repeat=5)
        one = t.Timer(1, interval=0, repeat=5)
        five2 = t.Timer(5, interval=0, repeat=5)

        neg_rand = t.Negate(t.Share(rand))

        x1 = t.Add(rand, five)  # 5 + rand
        x2 = t.Sub(x1, five2)  # rand
        x3 = t.Add(x2, neg_rand)  # 0
        res2 = t.Add(x3, one)  # 1
        p2 = t.Print(res2)  # 1

        t.PPrint(p2)
        t.run(p2)

    def test_3(self):
        print('''
        ******************************
        *           test3            *
        ******************************
        ''')

        def stream():
            for i in range(10):
                yield i

        f = t.Foo(stream)
        sum = t.Sum(t.Share(f))
        count = t.Count(t.Share(f))
        f3 = t.Div(sum, count)
        p3 = t.Print(f3)

        t.PPrint(p3)
        t.run(p3)

    def test_4(self):
        print('''
        ******************************
        *           test4            *
        ******************************
        ''')

        def stream(state):
            for i in range(10):
                yield i + state.val

        f = t.Foo(t.State(stream, val=5))
        p4 = t.Print(f)

        t.GraphViz(p4, 'test4')
        t.run(p4)

    def test_5(self):
        print('''
        ******************************
        *           test5            *
        ******************************
        ''')

        def myfoo(state, data):
            state.count = state.count + 1
            data['count'] = state.count
            return data

        p5 = t.Print(t.Apply(t.State(myfoo, count=0), t.Random()))
        t.GraphViz(p5, 'test5')
        t.run(p5)

    def test_6(self):
        print('''
        ******************************
        *           test6            *
        ******************************
        ''')

        def ran():
            for i in range(10):
                yield i

        p6 = t.Print(t.Window(ran, size=3, full_only=True))
        t.GraphViz(p6, 'test6')
        t.run(p6)
