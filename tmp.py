import tributary as t
import random
import time


def foo():
    return random.random()


# def long():
#     print('long called')
#     time.sleep(.1)
#     return 5


# print('''
# ******************************
# *           test             *
# ******************************
# ''')
# test = t.Timer(foo, interval=.1, repeat=5)
# test2 = t.Negate(t.Share(test))
# res = test + test2
# p = t.Print(res)
# t.GraphViz(p, 'test1')
# t.run(p)


# print('''
# ******************************
# *           test2            *
# ******************************
# ''')


# def foo():
#     print('foo called')
#     time.sleep(.1)
#     return random.random()

# rand = t.Timer(foo, interval=.1, repeat=5)
# five = t.Timer(long, repeat=5)
# one = t.Timer(1, repeat=5)
# five2 = t.Timer(5, repeat=5)

# neg_rand = t.Negate(t.Share(rand))

# x1 = rand + five  # 5 + rand
# x2 = x1 - five2  # rand
# x3 = x2 + neg_rand  # 0
# res2 = x3 + one  # 1
# p2 = t.Print(res2)  # 1

# t.PPrint(p2)
# t.GraphViz(p2, 'test2')
# t.run(p2)


# print('''
# ******************************
# *           test3            *
# ******************************
# ''')


# def stream():
#     for i in range(10):
#         yield i

# f = t.Foo(stream)
# sum = t.Sum(t.Share(f))
# count = t.Count(t.Share(f))
# f3 = sum / count
# p3 = t.Print(f3)

# t.PPrint(p3)
# t.GraphViz(p3, 'test3')
# t.run(p3)

# print('''
# ******************************
# *           test4            *
# ******************************
# ''')


# def stream(state):
#     for i in range(10):
#         yield i + state.val

# f = t.Foo(t.State(stream, val=5))
# p4 = t.Print(f)

# t.GraphViz(p4, 'test4')
# t.run(p4)

# print('''
# ******************************
# *           test5            *
# ******************************
# ''')


# def myfoo(state, data):
#     state.count = state.count + 1
#     data['count'] = state.count
#     return data

# p5 = t.Print(t.Apply(t.State(myfoo, count=0), t.Random()))
# t.GraphViz(p5, 'test5')
# t.run(p5)

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
