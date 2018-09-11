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
        # print(self.foo, self._last)
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

        if hasattr(self.foo, '__wraps__'):
            key = self.foo.__name__ + str(_id)
            ret[key] = []

            for f in self.foo.__wraps__:
                if isinstance(f, FunctionWrapper):
                    ret[key].append(f.view(_id+1))
                else:
                    ret[key].append(str(f))
            return ret
        ret[self.foo.__name__] = []
        return ret

    def __call__(self):
        ret = self.foo(**self.foo_kwargs)
        if isinstance(ret, types.GeneratorType):
            for r in ret:
                if isinstance(r, types.FunctionType):
                    tmp = r()
                else:
                    tmp = r
                if isinstance(tmp, types.GeneratorType):
                    for rr in tmp:
                        self.last = rr
                        yield self.last
                else:
                    self.last = tmp
                    yield self.last
        else:
            if isinstance(ret, types.FunctionType):
                tmp = ret()
            else:
                tmp = ret
            if isinstance(tmp, types.GeneratorType):
                for rr in tmp:
                    self.last = rr
                    yield self.last
            else:
                self.last = tmp
                yield self.last

    def __iter__(self):
        c_gen = self.__call__()
        for c in c_gen:
            yield c
