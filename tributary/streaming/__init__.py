import asyncio
from .node import Node as StreamingNode  # noqa: F401
from .base import StreamingGraph  # noqa: F401
from .calculations import *  # noqa: F401, F403
from .control import *  # noqa: F401, F403
from .input import *  # noqa: F401, F403
from .output import *  # noqa: F401, F403
from .utils import *  # noqa: F401, F403
from ..base import StreamEnd


async def _run(node):
    out = Collect(node)  # noqa F405
    graph = out._construct_graph()
    nodes = graph.getNodes()

    value, last = None, None

    while True:
        for level in nodes:
            await asyncio.gather(*(asyncio.create_task(n()) for n in level))
        value, last = out.value(), value
        if isinstance(value, StreamEnd):
            break
    return last


def run(node):
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # return future
        return asyncio.create_task(_run(node))
    # block until done
    return loop.run_until_complete(_run(node))
