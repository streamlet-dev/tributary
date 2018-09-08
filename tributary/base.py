import threading
from copy import deepcopy
from abc import abstractmethod, ABCMeta
from future.utils import with_metaclass
from .hosts.comm import runComm
from .utils import _q_me, _get_session, _start_sender_thread

_LANTERN_LIVE_RANK = 0


class LanternLive(object):
    def __init__(self, queue, path, live_thread=None, qput=None):
        self._path = path
        self._queue = queue
        self._thread = live_thread
        self._pipeline = None

        # if not a Streaming
        if not self._thread:
            self.on_data = qput

    def load(self, data):
        self._queue.put(data)

    def path(self):
        return self._path

    def _setBaseFoo(self, l):
        self._pipeline = l

    def pipeline(self):
        return self._pipeline()

    def __repr__(self):
        return self._path

    def __del__(self):
        pass
        # self._thread.stop()
        # self._thread.join()


class Streaming(with_metaclass(ABCMeta)):
    @abstractmethod
    def run(self):
        pass

    def on_data(self, data):
        getattr(self, '_qput')(data)


def run(streamer, sleep=1):
    global _LANTERN_LIVE_RANK

    q, qput = _q_me()

    # TODO add secret
    sessionid = _get_session()

    # start comm sender thread
    _LANTERN_LIVE_RANK = _start_sender_thread(runComm, q, _LANTERN_LIVE_RANK, sleep)

    # start streamer thread
    streamer._qput = qput
    t2 = threading.Thread(target=streamer.run)
    t2.start()

    ll = LanternLive(q, 'comm://' + sessionid + '/' + 'lantern.live/' + str(_LANTERN_LIVE_RANK-1), t2, qput)
    return ll


def pipeline(foos, foo_callbacks, foo_kwargs=None, sleep=1):
    global _LANTERN_LIVE_RANK

    foo_kwargs = foo_kwargs or []

    q, qput = _q_me()

    # TODO add secret
    sessionid = _get_session()

    # start comm sender thread
    _LANTERN_LIVE_RANK = _start_sender_thread(runComm, q, _LANTERN_LIVE_RANK, sleep)

    ll = LanternLive(q, 'comm://' + sessionid + '/' + 'lantern.live/' + str(_LANTERN_LIVE_RANK-1), None, qput)

    # organize args for functional pipeline
    assembled = []
    for i, foo in enumerate(foos):
        cb = foo_callbacks[i] if i < len(foo_callbacks) else 'on_data'
        kwargs = foo_kwargs[i] if i < len(foo_kwargs) else {}
        assembled.append((foo, cb, kwargs))

    # assemble pipeline
    assembled.reverse()
    lambdas = [ll.on_data]
    for i, a in enumerate(assembled):
        foo, cb, kwargs = a
        kwargs[cb] = lambdas[i]

        if i != len(assembled)-1:
            lambdas.append(lambda d, kw=kwargs, f=foo: f(d, **kw))
            lambdas[-1].__name__ = 'lambda-%d' % i
        else:
            lambdas.append(lambda kw=kwargs, f=foo: f(**kw))
            lambdas[-1].__name__ = 'lambda-%d' % i

    # TODO run on thread?
    ll._setBaseFoo(lambdas[-1])

    t = threading.Thread(target=ll.pipeline)
    t.start()

    return ll
