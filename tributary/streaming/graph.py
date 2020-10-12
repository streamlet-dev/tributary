import asyncio
import sys
from threading import Thread
from ..base import StreamEnd, StreamNone, StreamRepeat  # noqa: F401


class _Graph(object):
    '''internal representation of the entire graph state'''

    def __init__(self, node):
        self._stop = False
        self._starting_node = node  # noqa F405
        self.getNodes()

    def getNodes(self):
        self._nodes = self._starting_node._deep_bfs()
        return self._nodes

    def rebuild(self):
        # TODO
        return self._nodes

    def stop(self):
        self._stop = True

    async def _run(self):
        value, last, self._stop = None, None, False

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

        return last

    def run(self, blocking=True):
        if sys.platform == 'win32':
            # Set to proactor event loop on window
            # (default in python 3.8+)
            loop = asyncio.ProactorEventLoop()
        else:
            loop = asyncio.get_event_loop()

        asyncio.set_event_loop(loop)

        if loop.is_running():
            # return future
            return asyncio.create_task(self._run())

        if blocking:
            # block until done
            return loop.run_until_complete(self._run())

        t = Thread(target=loop.run_until_complete, args=(self._run(),))
        t.daemon = True
        t.start()
        return loop
