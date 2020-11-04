from .node import Node as StreamingNode  # noqa: F401
from .graph import StreamingGraph  # noqa: F401
from .calculations import *  # noqa: F401, F403
from .control import *  # noqa: F401, F403
from .input import *  # noqa: F401, F403
from .output import *  # noqa: F401, F403
from .scheduler import Scheduler  # noqa: F401
from .utils import *  # noqa: F401, F403
from ..base import StreamEnd, StreamNone, StreamRepeat  # noqa: F401


def run(node, blocking=True):
    graph = node.constructGraph()
    return graph.run(blocking)
