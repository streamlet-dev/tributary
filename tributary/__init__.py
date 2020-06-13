from gevent import monkey
_PATCHED = False

if not _PATCHED:
    monkey.patch_all(thread=False, select=False)
    _PATCHED = True


from ._version import __version__  # noqa: F401, E402

from .lazy import LazyGraph, LazyNode, node  # noqa: F401, E402
from .functional import pipeline, stop  # noqa: F401, E402
from .streaming import *  # noqa: F401, F403, E402
from .utils import LazyToStreaming  # noqa: F401, E402
