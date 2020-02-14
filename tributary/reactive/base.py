import types
import sys
from six import iteritems
DEBUG = False


def _wrap(foo, foo_kwargs, name='', wraps=(), share=None, state=None):
    '''wrap a function in a streaming variant

    Args:
        foo (callable): function to wrap
        foo_kwargs (dict): kwargs of function
        name (str): name of function to call
        wraps (tuple): functions or FunctionWrappers that this is wrapping
        share:
        state: state context

    Returns:
        FunctionWrapper: wrapped function
    '''
    if isinstance(foo, FunctionWrapper):
        ret = foo
    else:
        ret = FunctionWrapper(foo, foo_kwargs, name, wraps, share, state)

    for wrap in wraps:
        if isinstance(wrap, FunctionWrapper):
            if wrap._foo == ret._foo:
                continue
            _inc_ref(wrap, ret)
    return ret


def _inc_ref(f_wrapped, f_wrapping):
    '''Increment reference count for wrapped f

    Args:
        f_wrapped (FunctionWrapper): function that is wrapped
        f_wrapping (FunctionWrapper): function that wants to use f_wrapped
    '''
    if f_wrapped._id == f_wrapping._id:
        raise Exception('Internal Error')

    if f_wrapped._using is None:
        f_wrapped._using = id(f_wrapping)
        return
    Share(f_wrapped)


def Const(val):
    '''Streaming wrapper around scalar val

    Arguments:
        val (any): a scalar
    Returns:
        FunctionWrapper: a streaming wrapper
    '''
    return _wrap(val, dict(), name='Const', wraps=(val,))


def Foo(foo, foo_kwargs=None):
    '''Streaming wrapper around function call

    Arguments:
        foo (callable): a function or callable
        foo_kwargs (dict): kwargs for the function or callable foo
    Returns:
        FunctionWrapper: a streaming wrapper around foo
    '''
    return _wrap(foo, foo_kwargs or {}, name='Foo', wraps=(foo,))


def Share(f_wrap):
    '''Function to increment dataflow node reference count

    Arguments:
        f_wrap (FunctionWrapper): a streaming function
    Returns:
        FunctionWrapper: the same
    '''
    if not isinstance(f_wrap, FunctionWrapper):
        raise Exception('Share expects a tributary')
    f_wrap.inc()
    return f_wrap


class FunctionWrapper(object):
    '''Generic streaming wrapper for a function'''
    _id_ref = 0

    def __init__(self, foo, foo_kwargs, name='', wraps=(), share=None, state=None):
        '''
            Args:
        foo (callable): function to wrap
        foo_kwargs (dict): kwargs of function
        name (str): name of function to call
        wraps (tuple): functions or FunctionWrappers that this is wrapping
        share:
        state: state context

    Returns:
        FunctionWrapper: wrapped function

        '''
        self._id = FunctionWrapper._id_ref
        FunctionWrapper._id_ref += 1
        state = state or {}

        if not (isinstance(foo, types.FunctionType) or isinstance(foo, types.CoroutineType)):
            # bind to f so foo isnt referenced
            def _always(val=foo):
                while True:
                    yield val
            foo = _always

        if len(foo.__code__.co_varnames) > 0 and \
           foo.__code__.co_varnames[0] == 'state':
            self._foo = foo.__get__(self, FunctionWrapper)  # TODO: remember what this line does
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
        '''Get last call value'''
        if not hasattr(self, '_last'):
            raise Exception('Never called!!')

        if self._refs < 0:
            raise Exception('Ref mismatch in %s' % str(self._foo))

        self._refs -= 1
        return self._last

    def set_last(self, val):
        '''Set last call value'''
        self._refs = self._refs_orig
        self._last = val

    last = property(get_last, set_last)

    def inc(self):
        '''incremenet reference count'''
        self._refs_orig += 1
        self._refs += 1

    def view(self, _id=0, _idmap=None):
        '''Return tree representation of data stream'''
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

    async def __call__(self, *args, **kwargs):
        if DEBUG:
            print("calling: {}".format(self._foo))

        kwargs.update(self._foo_kwargs)

        async for item in _extract(self._foo, *args, **kwargs):
            self.last = item
            # while 0 < self._refs <= self._refs_orig:
            yield self.last

    def __iter__(self):
        yield from self.__call__()


async def _extract(item, *args, **kwargs):
    while isinstance(item, FunctionWrapper) or isinstance(item, types.FunctionType) or isinstance(item, types.CoroutineType):
        if isinstance(item, FunctionWrapper):
            item = item()

        if isinstance(item, types.FunctionType):
            item = item(*args, **kwargs)

        if isinstance(item, types.CoroutineType):
            item = await item

    if isinstance(item, types.AsyncGeneratorType):
        async for subitem in item:
            async for extracted in _extract(subitem):
                yield extracted

    elif isinstance(item, types.GeneratorType):
        for subitem in item:
            async for extracted in _extract(subitem):
                yield extracted
    else:
        yield item
