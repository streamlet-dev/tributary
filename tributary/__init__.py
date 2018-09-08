import time
from concurrent.futures import ThreadPoolExecutor, _base
from concurrent.futures.thread import BrokenThreadPool, _shutdown, _WorkItem
from functools import partial

_EXECUTOR = ThreadPoolExecutor(max_workers=2)


def timer(foo, args=None, kwargs=None, interval=1, repeat=0):
    args = args or ()
    kwargs = kwargs or {}

    _p = partial(foo, *args, **kwargs)

    def _repeater(repeat=repeat, *args, **kwargs):
        while repeat > 0:
            yield _p(*args, **kwargs)
            repeat -= 1
            time.sleep(interval)

    return _repeater


''' Python concurrent future
    def submit(self, fn, *args, **kwargs):
        with self._shutdown_lock:
            if self._broken:
                raise BrokenThreadPool(self._broken)

            if self._shutdown:
                raise RuntimeError('cannot schedule new futures after shutdown')
            if _shutdown:
                raise RuntimeError('cannot schedule new futures after'
                                   'interpreter shutdown')

            f = _base.Future()
            w = _WorkItem(f, fn, args, kwargs)

            self._work_queue.put(w)
            self._adjust_thread_count()
            return f
'''


def submit(fn, *args, **kwargs):
    self = _EXECUTOR
    with self._shutdown_lock:
        if self._broken:
            raise BrokenThreadPool(self._broken)

        if self._shutdown:
            raise RuntimeError('cannot schedule new futures after shutdown')
        if _shutdown:
            raise RuntimeError('cannot schedule new futures after'
                               'interpreter shutdown')

        f = _base.Future()
        w = _WorkItem(f, fn, args, kwargs)

        self._work_queue.put(w)
        self._adjust_thread_count()
        return f


def run_submit(fn, *args, **kwargs):
    f = submit(fn, *args, **kwargs)
    return f.result()


def pipeline(foos, foo_callbacks, foo_kwargs=None, sleep=1, on_data=print):
    foo_kwargs = foo_kwargs or []

    # organize args for functional pipeline
    assembled = []
    for i, foo in enumerate(foos):
        cb = foo_callbacks[i] if i < len(foo_callbacks) else 'on_data'
        kwargs = foo_kwargs[i] if i < len(foo_kwargs) else {}
        assembled.append((foo, cb, kwargs))

    # assemble pipeline
    assembled.reverse()
    lambdas = [on_data]
    for i, a in enumerate(assembled):
        foo, cb, kwargs = a
        kwargs[cb] = lambdas[i]

        if i != len(assembled)-1:
            # lambdas.append(lambda d, kw=kwargs, f=foo: f(d, **kw))
            lambdas.append(lambda d, kw=kwargs, f=foo: run_submit(f, d, **kw))
            lambdas[-1].__name__ = 'lambda-%d' % i
        else:
            # lambdas.append(lambda kw=kwargs, f=foo: f(**kw))
            lambdas.append(lambda kw=kwargs, f=foo: run_submit(f, **kw))
            lambdas[-1].__name__ = 'lambda-%d' % i

    # return entrypoint
    return lambdas[-1]
