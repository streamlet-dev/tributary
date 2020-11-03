import aiofiles
import json as JSON
from .input import Foo


class File(Foo):
    '''Open up a file and yield back lines in the file

    Args:
        filename (str): filename to read
        json (bool): load file line as json
    '''

    def __init__(self, filename, json=True):
        async def _file(filename=filename, json=json):
            async with aiofiles.open(filename) as f:
                async for line in f:
                    if json:
                        yield JSON.loads(line)
                    else:
                        yield line
        super().__init__(foo=_file)
        self._name = 'File'
