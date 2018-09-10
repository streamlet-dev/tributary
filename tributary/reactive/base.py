import types


def _wrap(foo, foo_kwargs):
    if isinstance(foo, FunctionWrapper):
        return foo
    return FunctionWrapper(foo, foo_kwargs)


class FunctionWrapper(object):
    def __init__(self, foo, foo_kwargs):
        self.foo = foo
        self.foo_kwargs = foo_kwargs
        self._refs_orig, self._refs = 1, 1

    def get_last(self):
        if not hasattr(self, '_last'):
            raise Exception('Never called!!')
        if self._refs <= 0:
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

    def __call__(self):
        if self._refs != 0 and self._refs < self._refs_orig:
            yield self.last
        else:
            ret = self.foo(**self.foo_kwargs)
            if isinstance(ret, types.GeneratorType):
                for r in ret:
                    if isinstance(r, types.FunctionType):
                        self.last = r()
                    else:
                        self.last = r
                    yield self.last
            else:
                if isinstance(ret, types.FunctionType):
                    self.last = ret()
                else:
                    self.last = ret
                yield self.last

    def __iter__(self):
        c_gen = self.__call__()
        for c in c_gen:
            yield c
