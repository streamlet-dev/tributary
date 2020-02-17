import asyncio
from .base import Node
from ..base import StreamNone, StreamRepeat


class Delay(Node):
    '''Streaming wrapper to delay a stream

    Arguments:
        node (node): input stream
        delay (float): time to delay input stream
    '''

    def __init__(self, node, delay=1):
        async def foo(val):
            await asyncio.sleep(delay)
            return val

        super().__init__(foo=foo, name='Delay', inputs=1)
        node._downstream.append((self, 0))
        self._upstream.append(node)


# class State(Node):
#     '''Streaming wrapper to delay a stream

#     Arguments:
#         node (node): input stream
#         state (dict): state dictionary of values to hold
#     '''

#     def __init__(self, node, delay=1):
#         async def foo(val):
#             await asyncio.sleep(delay)
#             return val

#         super().__init__(foo=foo, foo_kwargs=None, name='Delay', inputs=1)
#         node._downstream.append((self, 0))
#         self._upstream.append(node)


class Apply(Node):
    '''Streaming wrapper to apply a function to an input stream

    Arguments:
        node (node): input stream
        foo (callable): function to apply
        foo_kwargs (dict): kwargs for function
    '''
    def __init__(self, node, foo, foo_kwargs=None):
        self._apply = foo
        self._apply_kwargs = foo_kwargs or {}

        async def foo(val):
            return self._apply(val, **self._apply_kwargs)

        super().__init__(foo=foo, name='Apply', inputs=1)
        node._downstream.append((self, 0))
        self._upstream.append(node)


class Window(Node):
    '''Streaming wrapper to collect a window of values

    Arguments:
        node (node): input stream
        size (int): size of windows to use
        full_only (bool): only return if list is full
    '''
    def __init__(self, node, size=-1, full_only=False):
        self._accum = []

        def foo(val, size=size, full_only=full_only):
            if size == 0:
                return val
            else:
                self._accum.append(val)

            if size > 0:
                self._accum = self._accum[-size:]

            if full_only and len(self._accum) == size:
                return self._accum
            elif full_only:
                return StreamNone()
            else:
                return self._accum

        super().__init__(foo=foo, name='Window', inputs=1)
        node._downstream.append((self, 0))
        self._upstream.append(node)


class Unroll(Node):
    '''Streaming wrapper to unroll an iterable stream

    Arguments:
        node (node): input stream
    '''
    def __init__(self, node):
        self._count = 0

        async def foo(value):
            print('foo got:', value)
            # unrolled
            if self._count > 0:
                self._count -= 1
                return value

            # unrolling
            try:
                for v in value:
                    self._count += 1
                    print('pushing:', v)
                    await self._push(v, 0)
            except TypeError:
                return value
            else:
                return StreamRepeat()

        super().__init__(foo=foo, name='Unroll', inputs=1)
        node._downstream.append((self, 0))
        self._upstream.append(node)

# class UnrollDataFrame(Node):
#     '''Streaming wrapper to unroll an iterable stream

#     Arguments:
#         node (node): input stream
#     '''
#     def __init__(self, node, json=False, wrap=False):
#         def foo(value, json=json, wrap=wrap):
#             for i in range(len(value)):
#                 row = df.iloc[i]
#                 if json:
#                     data = row.to_dict()
#                     data['index'] = row.name
#                     yield data
#                 else:
#                     yield row

#         super().__init__(foo=foo, name='Unroll', inputs=1)
#         node._downstream.append((self, 0))
#         self._upstream.append(node)


# def Merge(f_wrap1, f_wrap2):
#     if not isinstance(f_wrap1, FunctionWrapper):
#         if not isinstance(f_wrap1, types.FunctionType):
#             f_wrap1 = Const(f_wrap1)
#         else:
#             f_wrap1 = Foo(f_wrap1)

#     if not isinstance(f_wrap2, FunctionWrapper):
#         if not isinstance(f_wrap2, types.FunctionType):
#             f_wrap2 = Const(f_wrap2)
#         else:
#             f_wrap2 = Foo(f_wrap2)

#     async def _merge(foo1, foo2):
#         async for gen1, gen2 in zip(foo1(), foo2()):
#             if isinstance(gen1, types.AsyncGeneratorType) and \
#                isinstance(gen2, types.AsyncGeneratorType):
#                 async for f1, f2 in zip(gen1, gen2):
#                     yield [f1, f2]
#             elif isinstance(gen1, types.AsyncGeneratorType):
#                 async for f1 in gen1:
#                     yield [f1, gen2]
#             elif isinstance(gen2, types.AsyncGeneratorType):
#                 async for f2 in gen2:
#                     yield [gen1, f2]
#             else:
#                 yield [gen1, gen2]

#     return _wrap(_merge, dict(foo1=f_wrap1, foo2=f_wrap2), name='Merge', wraps=(f_wrap1, f_wrap2), share=None)


# def ListMerge(f_wrap1, f_wrap2):
#     if not isinstance(f_wrap1, FunctionWrapper):
#         if not isinstance(f_wrap1, types.FunctionType):
#             f_wrap1 = Const(f_wrap1)
#         else:
#             f_wrap1 = Foo(f_wrap1)

#     if not isinstance(f_wrap2, FunctionWrapper):
#         if not isinstance(f_wrap2, types.FunctionType):
#             f_wrap2 = Const(f_wrap2)
#         else:
#             f_wrap2 = Foo(f_wrap2)

#     async def _merge(foo1, foo2):
#         async for gen1, gen2 in zip(foo1(), foo2()):
#             if isinstance(gen1, types.AsyncGeneratorType) and \
#                isinstance(gen2, types.AsyncGeneratorType):
#                 async for f1, f2 in zip(gen1, gen2):
#                     ret = []
#                     ret.extend(f1)
#                     ret.extend(f1)
#                     yield ret
#             elif isinstance(gen1, types.AsyncGeneratorType):
#                 async for f1 in gen1:
#                     ret = []
#                     ret.extend(f1)
#                     ret.extend(gen2)
#                     yield ret
#             elif isinstance(gen2, types.AsyncGeneratorType):
#                 async for f2 in gen2:
#                     ret = []
#                     ret.extend(gen1)
#                     ret.extend(f2)
#                     yield ret
#             else:
#                 ret = []
#                 ret.extend(gen1)
#                 ret.extend(gen2)
#                 yield ret

#     return _wrap(_merge, dict(foo1=f_wrap1, foo2=f_wrap2), name='ListMerge', wraps=(f_wrap1, f_wrap2), share=None)


# def DictMerge(f_wrap1, f_wrap2):
#     if not isinstance(f_wrap1, FunctionWrapper):
#         if not isinstance(f_wrap1, types.FunctionType):
#             f_wrap1 = Const(f_wrap1)
#         else:
#             f_wrap1 = Foo(f_wrap1)

#     if not isinstance(f_wrap2, FunctionWrapper):
#         if not isinstance(f_wrap2, types.FunctionType):
#             f_wrap2 = Const(f_wrap2)
#         else:
#             f_wrap2 = Foo(f_wrap2)

#     async def _dictmerge(foo1, foo2):
#         async for gen1, gen2 in zip(foo1(), foo2()):
#             if isinstance(gen1, types.AsyncGeneratorType) and \
#                isinstance(gen2, types.AsyncGeneratorType):
#                 async for f1, f2 in zip(gen1, gen2):
#                     ret = {}
#                     ret.update(f1)
#                     ret.update(f1)
#                     yield ret
#             elif isinstance(gen1, types.AsyncGeneratorType):
#                 async for f1 in gen1:
#                     ret = {}
#                     ret.update(f1)
#                     ret.update(gen2)
#                     yield ret
#             elif isinstance(gen2, types.AsyncGeneratorType):
#                 async for f2 in gen2:
#                     ret = {}
#                     ret.update(gen1)
#                     ret.update(f2)
#                     yield ret
#             else:
#                 ret = {}
#                 ret.update(gen1)
#                 ret.update(gen2)
#                 yield ret

#     return _wrap(_dictmerge, dict(foo1=f_wrap1, foo2=f_wrap2), name='DictMerge', wraps=(f_wrap1, f_wrap2), share=None)


# def Reduce(*f_wraps):
#     f_wraps = list(f_wraps)
#     for i, f_wrap in enumerate(f_wraps):
#         if not isinstance(f_wrap, types.FunctionType):
#             f_wraps[i] = Const(f_wrap)
#         else:
#             f_wraps[i] = Foo(f_wrap)

#     async def _reduce(foos):
#         async for all_gens in zip(*[foo() for foo in foos]):
#             gens = []
#             vals = []
#             for gen in all_gens:
#                 if isinstance(gen, types.AsyncGeneratorType):
#                     gens.append(gen)
#                 else:
#                     vals.append(gen)
#             if gens:
#                 for gens in zip(*gens):
#                     ret = list(vals)
#                     for gen in gens:
#                         ret.append(next(gen))
#                     yield ret
#             else:
#                 yield vals

#     return _wrap(_reduce, dict(foos=f_wraps), name='Reduce', wraps=tuple(f_wraps), share=None)
