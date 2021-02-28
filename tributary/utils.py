import functools
import inspect

import numpy as np
import pandas as pd

from .base import StreamEnd


def _either_type(f):
    """Utility decorator to allow for either no-arg decorator or arg decorator

    Args:
        f (callable): Callable to decorate
    """

    @functools.wraps(f)
    def new_dec(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            # actual decorated function
            return f(args[0])
        else:
            # decorator arguments
            return lambda realf: f(realf, *args, **kwargs)

    return new_dec


def LazyToStreaming(lazy_node):
    from .base import TributaryException
    from .lazy import LazyNode
    from .streaming import Foo, StreamingNode

    if isinstance(lazy_node, StreamingNode):
        return lazy_node
    if not isinstance(lazy_node, LazyNode):
        raise TributaryException("Malformed input:{}".format(lazy_node))

    return Foo(foo=lambda node=lazy_node: node())


def _compare(new_value, old_value):
    if isinstance(new_value, (int, float)) and type(new_value) == type(old_value):
        # if numeric, compare within a threshold
        # TODO
        return abs(new_value - old_value) < 0.00001

    elif type(new_value) != type(old_value):
        return False

    elif isinstance(new_value, (pd.DataFrame, pd.Series, np.ndarray)) or isinstance(
        old_value, (pd.DataFrame, pd.Series, np.ndarray)
    ):
        return (abs(new_value - old_value) < 0.00001).all()

    return new_value == old_value


def _ismethod(callable):
    return callable and (
        inspect.ismethod(callable)
        or (
            inspect.getargspec(callable).args
            and inspect.getargspec(callable).args[0] == "self"
        )
    )


def anext(obj):
    return obj.__anext__()


def _gen_to_foo(generator):
    try:
        return next(generator)
    except StopIteration:
        return StreamEnd()


async def _agen_to_foo(generator):
    try:
        return await anext(generator)
    except StopAsyncIteration:
        return StreamEnd()


def _gen_node(n):
    from .streaming import Const, Foo
    from .lazy import Node as LazyNode
    from .streaming import Node as StreamingNode

    if isinstance(n, StreamingNode):
        return n
    elif isinstance(n, LazyNode):
        return LazyToStreaming(n)
    elif callable(n):
        return Foo(n, name="Callable")
    return Const(n)
