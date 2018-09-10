from .base import _wrap
from .utils import Const, Timer, Print, Share
from .ops import Noop, Negate, Invert, Not, Add, Sub, Mult, Div, Mod, Pow, And, Or


def run(foo, **kwargs):
    foo = _wrap(foo, kwargs)
    for item in foo():
        pass
