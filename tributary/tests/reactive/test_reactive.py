import tributary as t
import random
import time


class TestConfig:
    def test_1(self):
        def foo():
            return random.random()

        print('''
        ******************************
        *           test             *
        ******************************
        ''')
        test = t.Timer(foo, {}, 0, 5)
        test2 = t.Negate(test)
        res2 = test + test2
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

        neg_rand = t.Negate(rand)

        x1 = rand + five  # 5 + rand
        x2 = x1 - five2  # rand
        x3 = x2 + neg_rand  # 0
        res2 = x3 + one  # 1
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
        sum = t.Sum(f)
        count = t.Count(f)
        f3 = sum / count
        p3 = t.Print(f3)

        t.PPrint(p3)
        t.run(p3)
