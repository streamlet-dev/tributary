import asyncio
from .base import Node  # noqa: F401
from .calculations import *  # noqa: F401, F403
from .input import *  # noqa: F401, F403
from .output import *  # noqa: F401, F403
from .utils import *  # noqa: F401, F403
from ..base import StreamEnd, StreamNone


async def _run(node):
    ret = []
    nodes = node._deep_bfs()
    while True:
        for level in nodes:
            await asyncio.gather(*(asyncio.create_task(n()) for n in level))
        if not isinstance(node.value(), (StreamEnd, StreamNone)):
            ret.append(node.value())
        elif isinstance(node.value(), StreamEnd):
            break
    return ret


def run(node):
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # return future
        return asyncio.create_task(_run(node))
    # block until done
    return loop.run_until_complete(_run(node))
