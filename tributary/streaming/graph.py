import asyncio
import sys
from threading import Thread

from ..base import StreamEnd, StreamNone, StreamRepeat, TributaryException  # noqa: F401


class StreamingGraph(object):
    """internal representation of the entire graph state"""

    def __init__(self, node):
        self._stop = False
        self._starting_node = node  # noqa F405

        # coroutines to run on start and stop
        self._onstarts = []
        self._onstops = []

        # Collect graph
        self.getNodes()

    def getNodes(self):
        self._nodes = self._starting_node._deep_bfs()

        # Run through nodes and extract onstarts and onstops
        for ns in self._nodes:
            for n in ns:
                if n._onstarts:
                    self._onstarts.extend(list(n._onstarts))
                if n._onstops:
                    self._onstops.extend(list(n._onstops))

        # Check that all are async coroutines
        for call in self._onstarts + self._onstops:
            if not asyncio.iscoroutinefunction(call):
                raise TributaryException(
                    "all onstarts and onstops must be async coroutines, got bad function: {}".format(
                        call
                    )
                )

        # return node levels
        return self._nodes

    def rebuild(self):
        # TODO
        return self._nodes

    def stop(self):
        self._stop = True

    async def _run(self):
        value, last, self._stop = None, None, False

        # run onstarts
        await asyncio.gather(*(asyncio.create_task(s()) for s in self._onstarts))

        while True:
            for level in self._nodes:
                if self._stop:
                    break

                await asyncio.gather(*(asyncio.create_task(n()) for n in level))

            self.rebuild()

            if self._stop:
                break

            value, last = self._starting_node.value(), value

            if isinstance(value, StreamEnd):
                break

        # run `onstops`
        await asyncio.gather(*(asyncio.create_task(s()) for s in self._onstops))

        # return last val
        return last

    def run(self, blocking=True, newloop=False, start=True):
        if sys.platform == "win32":
            # Set to proactor event loop on window
            # (default in python 3.8+)
            loop = asyncio.ProactorEventLoop()

        else:

            if newloop:
                loop = asyncio.new_event_loop()

            else:
                loop = asyncio.get_event_loop()

        asyncio.set_event_loop(loop)

        task = loop.create_task(self._run())

        if blocking:
            # block until done
            try:
                return loop.run_until_complete(task)
            except KeyboardInterrupt:
                return

        if start:
            t = Thread(target=loop.run_until_complete, args=(task,))
            t.daemon = True
            t.start()
            return loop

        return loop, task

    def graph(self):
        return self._starting_node.graph()

    def graphviz(self):
        return self._starting_node.graphviz()

    def dagre(self):
        return self._starting_node.dagre()
