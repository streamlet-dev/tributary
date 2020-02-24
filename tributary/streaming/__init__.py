import asyncio
from copy import deepcopy
from .base import StreamingGraph, Node as StreamingNode  # noqa: F401
from .calculations import *  # noqa: F401, F403
from .input import *  # noqa: F401, F403
from .output import *  # noqa: F401, F403
from .utils import *  # noqa: F401, F403
from ..base import StreamEnd, StreamNone, StreamRepeat


async def _run(node):
    ret = []
    nodes = node._deep_bfs()
    while True:
        for level in nodes:
            await asyncio.gather(*(asyncio.create_task(n()) for n in level))
        val = deepcopy(node.value())
        if not isinstance(val, (StreamEnd, StreamNone, StreamRepeat)):
            ret.append(val)
        elif isinstance(val, StreamEnd):
            break
    return ret


def run(node):
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # return future
        return asyncio.create_task(_run(node))
    # block until done
    return loop.run_until_complete(_run(node))
