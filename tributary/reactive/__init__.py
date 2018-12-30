from .base import _wrap, Foo, Const, Share  # noqa: F401
from .utils import *  # noqa: F401, F403
from .calculations import *  # noqa: F401, F403
from .input import *  # noqa: F401, F403
from .output import *  # noqa: F401, F403


def run(foo, **kwargs):
    foo = _wrap(foo, kwargs)
    ret = []
    try:
        for item in foo():
            ret.append(item)
    except KeyboardInterrupt:
        print('Terminating...')
    return ret
