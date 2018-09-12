from .base import _wrap
from .utils import *
from .calculations import *
from .input import *
from .output import *


def run(foo, **kwargs):
    foo = _wrap(foo, kwargs)
    ret = []
    try:
        for item in foo():
            ret.append(item)
    except KeyboardInterrupt:
        print('Terminating...')
    return ret
