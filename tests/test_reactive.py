import tributary as t
import random
import time


class TestConfig:
    def setup(self):
        pass
        # setup() before each test method

    def test_1(self):
        def foo():
            # return random.random()
            return 1

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
            # return random.random()
            return 1

        def long():
            time.sleep(0)
            return 5

        rand = t.Timer(foo, kwargs={}, interval=0, repeat=5)
        # rand2 = t.Timer(foo, kwargs={}, interval=0, repeat=5)
        five = t.Timer(long, kwargs={}, interval=0, repeat=5)
        one = t.Timer(1, kwargs={}, interval=0, repeat=5)
        five2 = t.Timer(5, {}, 0, 5)

        neg_rand = t.Negate(t.Share(rand))

        x1 = t.Add(rand, five)  # 5 + rand
        x2 = t.Sub(x1, five2)  # rand
        x3 = t.Add(x2, neg_rand)  # 0
        res = t.Add(x3, one)  # 1
        p = t.Print(res)  # 1

        t.PPrint(p)
        t.run(p)
