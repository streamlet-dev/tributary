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


def Window(node, size=-1, full_only=False):
    '''Streaming wrapper to collect a window of values

    Arguments:
        node (node): input stream
        size (int): size of windows to use
        full_only (bool): only return if list is full
    '''
    node._accum = []

    def foo(val, size=size, full_only=full_only):
        if size == 0:
            return val
        else:
            node._accum.append(val)

        if size > 0:
            node._accum = node._accum[-size:]

        if full_only and len(node._accum) == size:
            return node._accum
        elif full_only:
            return StreamNone()
        else:
            return node._accum
    
    ret = Node(foo=foo, name='Window', inputs=1)
    node._downstream.append((ret, 0))
    ret._upstream.append(node)
    return ret


class Unroll(Node):
    '''Streaming wrapper to unroll an iterable stream

    Arguments:
        node (node): input stream
    '''
    def __init__(self, node):
        self._count = 0

        async def foo(value):
            # unrolled
            if self._count > 0:
                self._count -= 1
                return value

            # unrolling
            try:
                for v in value:
                    self._count += 1
                    await self._push(v, 0)
            except TypeError:
                return value
            else:
                return StreamRepeat()

        super().__init__(foo=foo, name='Unroll', inputs=1)
        node._downstream.append((self, 0))
        self._upstream.append(node)


class UnrollDataFrame(Node):
    '''Streaming wrapper to unroll a dataframe into a stream

    Arguments:
        node (node): input stream
    '''
    def __init__(self, node, json=False, wrap=False):
        self._count = 0

        async def foo(value, json=json, wrap=wrap):
            # unrolled
            if self._count > 0:
                self._count -= 1
                return value

            # unrolling
            try:
                for i in range(len(value)):
                    row = value.iloc[i]

                    if json:
                        data = row.to_dict()
                        data['index'] = row.name
                    else:
                        data = row
                    self._count += 1
                    await self._push(data, 0)

            except TypeError:
                return value
            else:
                return StreamRepeat()

        super().__init__(foo=foo, name='UnrollDF', inputs=1)
        node._downstream.append((self, 0))
        self._upstream.append(node)


class Merge(Node):
    '''Streaming wrapper to merge 2 inputs into a single output

    Arguments:
        node1 (node): input stream
        node2 (node): input stream
    '''
    def __init__(self, node1, node2):
        def foo(value1, value2):
            return value1, value2

        super().__init__(foo=foo, name='Merge', inputs=2)
        node1._downstream.append((self, 0))
        node2._downstream.append((self, 1))
        self._upstream.append(node1)
        self._upstream.append(node2)


class ListMerge(Node):
    '''Streaming wrapper to merge 2 input lists into a single output list

    Arguments:
        node1 (node): input stream
        node2 (node): input stream
    '''
    def __init__(self, node1, node2):
        def foo(value1, value2):
            return list(value1) + list(value2)

        super().__init__(foo=foo, name='ListMerge', inputs=2)
        node1._downstream.append((self, 0))
        node2._downstream.append((self, 1))
        self._upstream.append(node1)
        self._upstream.append(node2)


class DictMerge(Node):
    '''Streaming wrapper to merge 2 input dicts into a single output dict.
       Preference is given to the second input (e.g. if keys overlap)

    Arguments:
        node1 (node): input stream
        node2 (node): input stream
    '''
    def __init__(self, node1, node2):
        def foo(value1, value2):
            ret = {}
            ret.update(value1)
            ret.update(value2)
            return ret

        super().__init__(foo=foo, name='DictMerge', inputs=2)
        node1._downstream.append((self, 0))
        node2._downstream.append((self, 1))
        self._upstream.append(node1)
        self._upstream.append(node2)


class Reduce(Node):
    '''Streaming wrapper to merge any number of inputs

    Arguments:
        nodes (tuple): input streams
    '''
    def __init__(self, *nodes):
        def foo(*values):
            return values

        super().__init__(foo=foo, name='Reduce', inputs=len(nodes))
        for i, n in enumerate(nodes):
            n._downstream.append((self, i))
            self._upstream.append(n)


Node.delay = Delay
# Node.state = State
Node.apply = Apply
Node.window = Window
Node.unroll = Unroll
Node.unrollDataFrame = UnrollDataFrame
Node.merge = Merge
Node.listMerge = ListMerge
Node.dictMerge = DictMerge
Node.reduce = Reduce
