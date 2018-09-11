import time
import types
from pprint import pprint
from .base import _wrap, FunctionWrapper

def Const(val):
    def _always(val):
        yield val

    _always.__name__ = 'Const'
    _always.__wraps__ = (val,)

    return _wrap(_always, dict(val=val))


def Timer(foo_or_val, kwargs=None, interval=1, repeat=0):
    kwargs = kwargs or {}

    if not isinstance(foo_or_val, types.FunctionType):
        _p = Const(foo_or_val)
    else:
        _p = _wrap(foo_or_val, kwargs)

    def _repeater(foo, repeat, interval):
        while repeat > 0:
            t1 = time.time()
            yield foo()
            t2 = time.time()

            if interval > 0:
                # sleep for rest of time that _p didnt take
                print('sleep', t2-t1-interval)
                time.sleep(max(0, t2-t1-interval))
            repeat -= 1

    _repeater.__name__ = 'Timer'
    _repeater.__wraps__ = (_p,)

    return _wrap(_repeater, dict(foo=_p, repeat=repeat, interval=interval))


def Print(foo, foo_kwargs=None):
    foo_kwargs = foo_kwargs or {}
    foo = _wrap(foo, foo_kwargs)

    def _print(foo):
        for r in foo():
            print(r)

    _print.__name__ = 'Print'
    _print.__wraps__ = (foo,)

    return _wrap(_print, dict(foo=foo))


def Share(f_wrap):
    if not isinstance(f_wrap, FunctionWrapper):
        raise Exception('Share expects tributary')
    f_wrap.inc()
    return f_wrap


def Graph(f_wrap):
    if not isinstance(f_wrap, FunctionWrapper):
        raise Exception('ViewGraph expects tributary')
    return f_wrap.view(0)


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
