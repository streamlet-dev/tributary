import aiofiles
import json as JSON
from ..base import _wrap  # noqa: F401


def File(foo, foo_kwargs=None, filename='', json=True):
    foo_kwargs = foo_kwargs or {}
    foo = _wrap(foo, foo_kwargs)

    async def _file(foo, filename, json):
        async for data in foo():
            async with aiofiles.open(filename) as fp:
                if json:
                    fp.write(JSON.dumps(data))
                else:
                    fp.write(data)

    return _wrap(_file, dict(foo=foo, filename=filename, json=json), name='File')
