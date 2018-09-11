import time
import types
from pprint import pprint
from .base import _wrap, FunctionWrapper


def Const(val):
    def _always(val):
        yield val

    return _wrap(_always, dict(val=val), name='Const', wraps=(val,))


def Foo(foo, foo_kwargs):
    return _wrap(foo, foo_kwargs, name='Foo', wraps=(foo,))


def Timer(foo_or_val, kwargs=None, interval=1, repeat=0):
    kwargs = kwargs or {}

    if not isinstance(foo_or_val, types.FunctionType):
        foo = Const(foo_or_val)
    else:
        foo = Foo(foo_or_val, kwargs)

    def _repeater(foo, repeat, interval):
        print('timer called')
        while repeat > 0:
            t1 = time.time()
            yield foo()
            t2 = time.time()

            if interval > 0:
                # sleep for rest of time that _p didnt take
                print('sleep', interval-(t2-t1))
                time.sleep(max(0, interval-(t2-t1)))
            repeat -= 1

    return _wrap(_repeater, dict(foo=foo, repeat=repeat, interval=interval), name='Timer', wraps=(foo,), share=foo)


def Print(foo, foo_kwargs=None):
    foo_kwargs = foo_kwargs or {}
    foo = _wrap(foo, foo_kwargs)

    def _print(foo):
        print('print called')
        for r in foo():
            print(r)

    return _wrap(_print, dict(foo=foo), name='Print', wraps=(foo,), share=foo)


def Share(f_wrap):
    if not isinstance(f_wrap, FunctionWrapper):
        raise Exception('Share expects tributary')
    f_wrap.inc()
    return f_wrap


def Graph(f_wrap):
    if not isinstance(f_wrap, FunctionWrapper):
        raise Exception('ViewGraph expects tributary')
    return f_wrap.view(0)[0]


def PPrint(f_wrap):
    pprint(Graph(f_wrap))


def GraphViz(f_wrap, name='Graph'):
    d = Graph(f_wrap)
    from graphviz import Digraph
    dot = Digraph(name)
    dot.format = 'png'

    def rec(nodes, parent):
        for d in nodes:
            if not isinstance(d, dict):
                dot.node(d)
                dot.edge(d, parent)

            else:
                for k in d:
                    dot.node(k)
                    rec(d[k], k)
                    dot.edge(k, parent)

    for k in d:
        dot.node(k)
        rec(d[k], k)

    dot.render()
