import asyncio
import types
from .base import _wrap, Foo, Const, Share  # noqa: F401
from .calculations import *  # noqa: F401, F403
from .input import *  # noqa: F401, F403
from .output import *  # noqa: F401, F403
from .utils import *  # noqa: F401, F403


async def _run(foo, **kwargs):
    foo = _wrap(foo, kwargs)
    ret = []
    try:
        async for item in foo():
            if isinstance(item, types.AsyncGeneratorType):
                async for i in item:
                    ret.append(i)
            elif isinstance(item, types.CoroutineType):
                ret.append(await item)
            else:
                ret.append(item)

    except KeyboardInterrupt:
        print('Terminating...')
    return ret


def run(foo, **kwargs):
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # return future
        return asyncio.create_task(_run(foo, **kwargs))
    else:
        # block until done
        x = loop.run_until_complete(_run(foo, **kwargs))
    return x


class BaseClass(object):
    def __init__(self, run=None, *args, **kwargs):
        self._run = run

    def run(self, **kwargs):
        if not hasattr(self, "_run") or not self._run:
            raise Exception("Reactive class improperly constructed, did you forget a super()?")
        return run(self._run, **kwargs)

    def pprint(self):
        return PPrint(self._run)

    def graphviz(self):
        return GraphViz(self._run)
