import asyncio
import types
from aiostream.aiter_utils import anext
from asyncio import Queue, QueueEmpty as Empty
from ..base import StreamEnd, StreamNone


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


class Node(object):
    _id_ref = 0

    def __init__(self, foo, foo_kwargs=None, name=None, inputs=1):
        self._id = Node._id_ref
        Node._id_ref += 1

        self._name = '{}#{}'.format(name or self.__class__.__name__, self._id)
        self._input = [Queue() for _ in range(inputs)]
        self._active = [StreamNone() for _ in range(inputs)]
        self._downstream = []
        self._upstream = []

        self._foo = foo
        self._foo_kwargs = foo_kwargs or {}
        self._last = StreamNone()
        self._finished = False

    def __repr__(self):
        return '{}'.format(self._name)

    async def _push(self, inp, index):
        await self._input[index].put(inp)

    async def _execute(self):
        if asyncio.iscoroutine(self._foo):
            _last = await self._foo(*self._active, **self._foo_kwargs)
        elif isinstance(self._foo, types.FunctionType):
            _last = self._foo(*self._active, **self._foo_kwargs)
        else:
            raise Exception('Cannot use type:{}'.format(type(self._foo)))

        if isinstance(_last, types.AsyncGeneratorType):
            async def _foo(g=_last):
                return await _agen_to_foo(g)
            self._foo = _foo
            _last = await self._foo()

        elif isinstance(_last, types.GeneratorType):
            self._foo = lambda g=_last: _gen_to_foo(g)
            _last = self._foo()
        elif asyncio.iscoroutine(_last):
            _last = await _last

        self._last = _last
        await self._output(self._last)
        for i in range(len(self._active)):
            self._active[i] = StreamNone()

    async def _finish(self):
        self._finished = True
        self._last = StreamEnd()
        await self._output(self._last)

    async def __call__(self):
        if self._finished:
            return await self._finish()

        ready = True
        # iterate through inputs
        for i, inp in enumerate(self._input):
            # if input hasn't received value
            if isinstance(self._active[i], StreamNone):
                try:
                    # get from input queue
                    val = inp.get_nowait()

                    if isinstance(val, StreamEnd):
                        return await self._finish()

                    # set as active
                    self._active[i] = val
                except Empty:
                    # wait for value
                    ready = False

        if ready:
            # execute function
            return await self._execute()

    async def _output(self, ret):
        # if downstreams, output
        for down, i in self._downstream:
            await down._push(ret, i)
        return ret

    def _deep_bfs(self, reverse=True):
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
        return self._last
