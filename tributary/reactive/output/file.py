import aiofiles
import json as JSON
from ..base import _wrap  # noqa: F401


def File(foo, foo_kwargs=None, filename='', json=True):
    '''Open up a file and write lines to the file

    Args:
        foo (callable): input stream
        foo_kwargs (dict): kwargs for the input stream
        filename (str): filename to write
        json (bool): load file line as json
    '''
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
