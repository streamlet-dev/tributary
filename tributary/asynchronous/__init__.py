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
    x = loop.run_until_complete(_run(foo, **kwargs))
    return x
