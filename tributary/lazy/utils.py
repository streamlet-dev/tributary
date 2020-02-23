from temporalcache import interval, expire
from .base import Node


def Expire(node, second=None, minute=None, hour=None, day=None, day_of_week=None, week=None, month=None, maxsize=128):
    def _interval(node=node):
        return node()

    # make new node
    ret = node._gennode('Expire[{}-{}-{}-{}-{}-{}-{}-maxsize:{}](-'.format(second, minute, hour, day, day_of_week, week, month, maxsize) + node._name + ')', _interval, [node], node._trace)

    # stash original recompute
    ret._orig_recompute = ret._recompute

    # make recompute run on expire
    ret._recompute = expire(second=second, minute=minute, hour=hour, day=day, day_of_week=day_of_week, week=week, month=month, maxsize=maxsize)(ret._recompute)
    return ret


def Interval(node, seconds=0, minutes=0, hours=0, days=0, weeks=0, months=0, years=0, maxsize=128):
    def _interval(node=node):
        return node()

    # make new node
    ret = node._gennode('Interval[{}-{}-{}-{}-{}-{}-{}-maxsize:{}'.format(seconds, minutes, hours, days, weeks, months, years, maxsize) + node._name + ')', _interval, [node], node._trace)

    # stash original recompute
    ret._orig_recompute = ret._recompute

    # make recompute run on interval
    ret._recompute = interval(seconds=seconds, minutes=minutes, hours=hours, days=days, weeks=weeks, months=months, years=years, maxsize=maxsize)(ret._recompute)
    return ret


Node.expire = Expire
Node.interval = Interval
