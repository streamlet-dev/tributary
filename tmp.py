import tributary as t
import random
import time


def foo():
    return random.random()


def long():
    print('long called')
    time.sleep(1)
    return 5


print('''
******************************
*           test             *
******************************
''')
test = t.Timer(foo, interval=.5, repeat=5)
test2 = t.Negate(t.Share(test))
res = t.Add(test, test2)
p = t.Print(res)
t.GraphViz(p, 'test1')
t.run(p)


print('''
******************************
*           test2            *
******************************
''')
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
t.GraphViz(p2, 'test2')
t.run(p2)


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
t.GraphViz(p3, 'test3')
t.run(p3)
