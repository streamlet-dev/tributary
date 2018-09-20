import types
import sys
from six import iteritems


def _wrap(foo, foo_kwargs, name='', wraps=(), share=None, state=None):
    if isinstance(foo, FunctionWrapper):
        ret = foo
    else:
        ret = FunctionWrapper(foo, foo_kwargs, name, wraps, share, state)

    for wrap in wraps:
        if isinstance(wrap, FunctionWrapper):
            if wrap == foo:
                continue
            _inc_ref(wrap, ret)
    return ret


def _call_if_function(f):
    if isinstance(f, types.FunctionType):
        return f()
    return f


def _inc_ref(f_wrapped, f_wrapping):
    if f_wrapped == f_wrapping:
        raise Exception('Internal Error')

    if f_wrapped._using is None or f_wrapped._using == id(f_wrapping):
        f_wrapped._using = id(f_wrapping)
        return
    Share(f_wrapped)
    f_wrapped._using = id(f_wrapping)


class FunctionWrapper(object):
    def __init__(self, foo, foo_kwargs, name='', wraps=(), share=None, state=None):
        state = state or {}

        if len(foo.__code__.co_varnames) > 0 and \
           foo.__code__.co_varnames[0] == 'state':
            self._foo = foo.__get__(self, FunctionWrapper)
            for k, v in iteritems(state):
                if k not in ('_foo', '_foo_kwargs', '_refs_orig', '_name', '_wraps', '_share'):
                    setattr(self, k, v)
                else:
                    raise Exception('Reserved Word - %s' % k)
        else:
            self._foo = foo

        self._foo_kwargs = foo_kwargs
        self._refs_orig, self._refs = 1, 1

        self._name = name
        self._wraps = wraps
        self._using = None
        self._share = share if share else self

    def get_last(self):
        if not hasattr(self, '_last'):
            raise Exception('Never called!!')

        if self._refs < 0:
            raise Exception('Ref mismatch in %s' % str(self._foo))

        self._refs -= 1
        return self._last

    def set_last(self, val):
        self._refs = self._refs_orig
        self._last = val

    last = property(get_last, set_last)

    def inc(self):
        self._refs_orig += 1
        self._refs += 1

    def view(self, _id=0, _idmap=None):
        _idmap = _idmap or {}
        ret = {}

        # check if i exist already in graph
        if id(self) in _idmap:
            key = _idmap[id(self)]
        else:
            key = self._name + str(_id)
            # _id += 1
            _idmap[id(self)] = key
        _id += 1

        ret[key] = []
        for f in self._wraps:
            if isinstance(f, FunctionWrapper):
                r, m = f.view(_id, _idmap)
                ret[key].append(r)
                _id = m
            else:
                if 'pandas' in sys.modules:
                    import pandas as pd
                    if isinstance(f, pd.DataFrame) or isinstance(f, pd.Series):
                        # pprint
                        f = 'DataFrame'
                ret[key].append(str(f))
        return ret, _id

    def __call__(self, *args, **kwargs):
        while(self._refs == self._refs_orig):
            kwargs.update(self._foo_kwargs)
            ret = self._foo(*args, **kwargs)
            if isinstance(ret, types.GeneratorType):
                for r in ret:
                    tmp = _call_if_function(r)

                    if isinstance(tmp, types.GeneratorType):
                        for rr in tmp:
                            self.last = rr
                            yield self.last

                    else:
                        self.last = tmp
                        yield self.last
            else:
                tmp = _call_if_function(ret)

                if isinstance(tmp, types.GeneratorType):
                    for rr in tmp:
                        self.last = rr
                        yield self.last

                else:
                    self.last = tmp
                    yield self.last
        while(0 < self._refs < self._refs_orig):
            yield self.last

        # reset state to be called again
        self._refs = self._refs_orig

    def __iter__(self):
        c_gen = self.__call__()
        for c in c_gen:
            yield c

    def __add__(self, other):
        pass


def Const(val):
    def _always(val):
        yield val

    return _wrap(_always, dict(val=val), name='Const', wraps=(val,))


def Foo(foo, foo_kwargs=None):
    return _wrap(foo, foo_kwargs or {}, name='Foo', wraps=(foo,))


def Share(f_wrap):
    if not isinstance(f_wrap, FunctionWrapper):
        raise Exception('Share expects a tributary')
    f_wrap.inc()
    return f_wrap
