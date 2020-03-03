import asyncio
import types
from asyncio import Queue, QueueEmpty as Empty
from ..base import StreamEnd, StreamNone, StreamRepeat


def anext(obj):
    return obj.__anext__()


def _gen_to_foo(generator):
    try:
        return next(generator)
    except StopIteration:
        return StreamEnd()


async def _agen_to_foo(generator):
    try:
        return await anext(generator)
    except StopAsyncIteration:
        return StreamEnd()


def _gen_node(n):
    from .input import Const, Foo
    if isinstance(n, Node):
        return n
    elif callable(n):
        return Foo(n, name="Callable")
    return Const(n)


class Node(object):
    '''A representation of a node in the forward propogating graph.

    Args:
        foo (callable): the python callable to wrap in a forward propogating node, can be:
                            - function
                            - generator
                            - async function
                            - async generator
        foo_kwargs (dict): kwargs for the wrapped callables, should be static call-to-call
        name (str): name of the node
        inputs (int): number of upstream inputs
        kwargs (dict): extra kwargs:
                        - delay_interval (int/float): rate limit
                        - execution_max (int): max number of times to execute callable
    '''
    _id_ref = 0

    def __init__(self, foo, foo_kwargs=None, name=None, inputs=0, **kwargs):
        # Instances get an id but one id tracker for all nodes so we can
        # uniquely identify them
        # TODO different scheme
        self._id = Node._id_ref
        Node._id_ref += 1

        # Graphviz shape
        self._graphvizshape = kwargs.get('graphvizshape', 'ellipse')

        # dagred3 node if live updating
        self._dd3g = None

        # Every node gets a name so it can be uniquely identified in the graph
        self._name = '{}#{}'.format(name or self.__class__.__name__, self._id)

        # Inputs are async queues from upstream nodes
        self._input = [Queue() for _ in range(inputs)]

        # Active are currently valid inputs, since inputs
        # may come at different rates
        self._active = [StreamNone() for _ in range(inputs)]

        # Downstream nodes so we can traverse graph, push
        # results to downstream nodes
        self._downstream = []

        # Upstream nodes so we can traverse graph, plot and optimize
        self._upstream = []

        # The function we are wrapping, can be:
        #    - vanilla function
        #    - vanilla generator
        #    - async function
        #    - async generator
        self._foo = foo

        # Any kwargs necessary for the function.
        # These should be static call-to-call.
        self._foo_kwargs = foo_kwargs or {}

        # Delay between executions, useful for rate-limiting
        # default is no rate limiting
        self._delay_interval = kwargs.get('delay_interval', 0)

        # max number of times to execute callable
        self._execution_max = kwargs.get('execution_max', 0)

        # current execution count
        self._execution_count = 0

        # last value pushed downstream
        self._last = StreamNone()

        # stream is in a finished state, will only propogate StreamEnd instances
        self._finished = False

    def __repr__(self):
        return '{}'.format(self._name)

    async def _startdd3g(self):
        if self._dd3g:
            self._dd3g.setNode(self._name, tooltip=str(self._last), style='fill: #0f0')
            await asyncio.sleep(0.1)

    async def _waitdd3g(self):
        if self._dd3g:
            self._dd3g.setNode(self._name, tooltip=str(self._last), style='fill: #ff0')
            await asyncio.sleep(0.1)

    async def _finishdd3g(self):
        if self._dd3g:
            self._dd3g.setNode(self._name, tooltip=str(self._last), style='fill: #f00')
            await asyncio.sleep(0.1)

    async def _enddd3g(self):
        if self._dd3g:
            self._dd3g.setNode(self._name, tooltip=str(self._last), style='fill: #fff')
            await asyncio.sleep(0.1)

    async def _push(self, inp, index):
        '''push value to downstream nodes'''
        await self._input[index].put(inp)

    async def _execute(self):
        '''execute callable'''
        valid = False
        while not valid:
            if asyncio.iscoroutine(self._foo):
                _last = await self._foo(*self._active, **self._foo_kwargs)
            elif isinstance(self._foo, types.FunctionType):
                try:
                    _last = self._foo(*self._active, **self._foo_kwargs)
                except ValueError:
                    # Swap back to function
                    self._foo = self._old_foo
                    continue
            else:
                raise Exception('Cannot use type:{}'.format(type(self._foo)))
            valid = True
            self._execution_count += 1

        if isinstance(_last, types.AsyncGeneratorType):
            async def _foo(g=_last):
                return await _agen_to_foo(g)
            self._foo = _foo
            _last = await self._foo()
        elif isinstance(_last, types.GeneratorType):
            # Swap to generator unroller
            self._old_foo = self._foo
            self._foo = lambda g=_last: _gen_to_foo(g)
            _last = self._foo()

        elif asyncio.iscoroutine(_last):
            _last = await _last

        self._last = _last
        await self._output(self._last)
        for i in range(len(self._active)):
            self._active[i] = StreamNone()
        await self._enddd3g()

    async def _finish(self):
        '''mark this node as finished'''
        self._finished = True
        self._last = StreamEnd()
        await self._finishdd3g()
        await self._output(self._last)

    def _backpressure(self):
        '''check if _downstream are all empty, if not then don't propogate'''
        ret = not all(n._input[i].empty() for n, i in self._downstream)
        return ret

    async def __call__(self):
        '''execute the callable if possible, and propogate values downstream'''
        # Downstream nodes can't process
        if self._backpressure():
            await self._waitdd3g()
            return StreamNone()

        # Previously ended stream
        if self._finished:
            return await self._finish()

        # dd3g
        await self._startdd3g()

        # Sleep if needed
        if self._delay_interval:
            await asyncio.sleep(self._delay_interval)

        # Stop executing
        if self._execution_max > 0 and self._execution_count >= self._execution_max:
            self._foo = lambda: StreamEnd()
            self._old_foo = lambda: StreamEnd()

        ready = True
        # iterate through inputs
        for i, inp in enumerate(self._input):
            # if input hasn't received value
            if isinstance(self._active[i], StreamNone):
                try:
                    # get from input queue
                    val = inp.get_nowait()

                    while isinstance(val, StreamRepeat):
                        # Skip entry
                        val = inp.get_nowait()

                    if isinstance(val, StreamEnd):
                        return await self._finish()

                    # set as active
                    self._active[i] = val

                except Empty:
                    # wait for value
                    self._active[i] = StreamNone()
                    ready = False

        if ready:
            # execute function
            return await self._execute()

    async def _output(self, ret):
        '''output value to downstream nodes'''
        # if downstreams, output
        if not isinstance(ret, (StreamNone, StreamRepeat)):
            for down, i in self._downstream:
                await down._push(ret, i)
        return ret

    def _deep_bfs(self, reverse=True):
        '''get nodes by level in tree, reversed relative to output node.
           e.g. given a tree that looks like:
        A -> B -> D -> F
         \\-> C -> E /
         the result will be: [[A], [B, C], [D, E], [F]]

         This will be the order we synchronously execute, so that within a
         level nodes' execution will be asynchronous but from level to level
         they will be synchronous
        '''
        nodes = []
        nodes.append([self])

        upstreams = self._upstream.copy()
        while upstreams:
            nodes.append(upstreams)
            upstreams = []
            for n in nodes[-1]:
                upstreams.extend(n._upstream)

        if reverse:
            nodes.reverse()

        return nodes

    def value(self):
        '''get value from node'''
        return self._last


class StreamingGraph(object):
    def __init__(self, output_node):
        self._node = output_node

    def graph(self):
        return self._node.graph()

    def graphviz(self):
        return self._node.graphviz()

    def dagre(self):
        return self._node.dagre()

    def run(self):
        from tributary.streaming import run
        return run(self._node)
