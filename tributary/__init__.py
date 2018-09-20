from gevent import monkey
_PATCHED = False
if not _PATCHED:
    monkey.patch_all(thread=False, select=False)
    _PATCHED = True


__version__ = '0.0.3'


from .functional import pipeline, stop
from .reactive import *