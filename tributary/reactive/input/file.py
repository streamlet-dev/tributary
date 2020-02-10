import aiofiles
import json as JSON
from ..base import _wrap  # noqa: F401


def File(filename, json=True):
    '''Open up a file and yield back lines in the file

    Args:
        filename (str): filename to read
        json (bool): load file line as json
    '''
    async def _file(filename, json):
        async with aiofiles.open(filename) as f:
            async for line in f:
                if json:
                    yield JSON.loads(line)
                else:
                    yield line

    return _wrap(_file, dict(filename=filename, json=json), name='File')
