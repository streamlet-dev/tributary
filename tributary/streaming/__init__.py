from .node import Node as StreamingNode  # noqa: F401
from .graph import StreamingGraph  # noqa: F401
from .calculations import *  # noqa: F401, F403
from .control import *  # noqa: F401, F403
from .input import *  # noqa: F401, F403
from .output import *  # noqa: F401, F403
from .utils import *  # noqa: F401, F403
from ..base import StreamEnd, StreamNone, StreamRepeat  # noqa: F401


def run(node, blocking=True, construct_only=False):
    graph = node._construct_graph()
    if construct_only:
        return graph
    return graph.run(blocking)
