import types


def _wrap(foo, foo_kwargs, name='', wraps=(), share=None):
    if isinstance(foo, FunctionWrapper):
        return foo
    return FunctionWrapper(foo, foo_kwargs, name, wraps, share)


def _call_if_function(f):
    if isinstance(f, types.FunctionType):
        return f()
    return f


class FunctionWrapper(object):
    def __init__(self, foo, foo_kwargs, name='', wraps=(), share=None):
        self.foo = foo
        self.foo_kwargs = foo_kwargs
        self._refs_orig, self._refs = 1, 1

        self.name = name
        self.wraps = wraps
        self.share = share if share else self

    def get_last(self):
        if not hasattr(self, '_last'):
            raise Exception('Never called!!')

        if self._refs < 0:
            raise Exception('Ref mismatch in %s' % str(self.foo))

        self._refs -= 1
        return self._last

    def set_last(self, val):
        self._refs = self._refs_orig
        self._last = val

    last = property(get_last, set_last)

    def inc(self):
        self._refs_orig += 1
        self._refs += 1

    def view(self, _id=0):
        ret = {}
        key = self.name + str(_id)
        ret[key] = []
        _id += 1

        for f in self.wraps:
            if isinstance(f, FunctionWrapper):
                r, m = f.view(_id)
                ret[key].append(r)
                _id = m
            else:
                ret[key].append(str(f))
        return ret, _id

    def __call__(self):
        while(self._refs == self._refs_orig):
            ret = self.foo(**self.foo_kwargs)
            # import ipdb; ipdb.set_trace()
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
