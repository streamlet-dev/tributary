from .base import StreamNone
from gevent import spawn


def run(target, timeout=1):
    last = None
    done = False

    while not done:
        g = spawn(target)
        g.join(timeout)

        while not g.successful():
            yield StreamNone(last)

        last = g.value
        if last is None:
            done = True
        else:
            yield last
