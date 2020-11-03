from temporalcache import interval, expire
from .base import Node


def Expire(
    node,
    second=None,
    minute=None,
    hour=None,
    day=None,
    day_of_week=None,
    week=None,
    month=None,
    maxsize=128,
):
    def _interval(node=node):
        return node()

    # make new node
    ret = node._gennode(
        "Expire[{}-{}-{}-{}-{}-{}-{}-maxsize:{}]({})".format(
            second, minute, hour, day, day_of_week, week, month, maxsize, node._name
        ),
        _interval,
        [node],
    )

    # stash original recompute
    ret._orig_recompute = ret._recompute

    # make recompute run on expire
    ret._recompute = expire(
        second=second,
        minute=minute,
        hour=hour,
        day=day,
        day_of_week=day_of_week,
        week=week,
        month=month,
        maxsize=maxsize,
    )(ret._recompute)
    return ret


def Interval(
    node, seconds=0, minutes=0, hours=0, days=0, weeks=0, months=0, years=0, maxsize=128
):
    def _interval(node=node):
        return node()

    # make new node
    ret = node._gennode(
        "Interval[{}-{}-{}-{}-{}-{}-{}-maxsize:{}]({})".format(
            seconds, minutes, hours, days, weeks, months, years, maxsize, node._name
        ),
        _interval,
        [node],
    )

    # stash original recompute
    ret._orig_recompute = ret._recompute

    # make recompute run on interval
    ret._recompute = interval(
        seconds=seconds,
        minutes=minutes,
        hours=hours,
        days=days,
        weeks=weeks,
        months=months,
        years=years,
        maxsize=maxsize,
    )(ret._recompute)
    return ret


def Window(node, size=-1, full_only=False):
    """Lazy wrapper to collect a window of values. If a node is executed 3 times,
    returning 1, 2, 3, then the window node will collect those values in a list.

    Arguments:
        node (node): input node
        size (int): size of windows to use
        full_only (bool): only return if list is full
    """

    def foo(node=node, size=size, full_only=full_only):
        if size == 0:
            return node.value()

        if ret._accum is None:
            ret._accum = [node.value()]

        elif ret.dependencyIsDirty(node):
            ret._accum.append(node.value())

        if size > 0:
            ret._accum = ret._accum[-size:]

        if full_only and len(ret._accum) == size:
            return ret._accum
        elif full_only:
            return None
        return ret._accum

    # make new node
    ret = node._gennode("Window[{}]".format(size if size > 0 else "∞"), foo, [node])
    ret._accum = None
    return ret


Node.expire = Expire
Node.interval = Interval
Node.window = Window
