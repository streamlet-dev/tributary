import aiofiles
import json as JSON
from ..base import Node


class File(Node):
    '''Open up a file and write lines to the file

    Args:
        node (Node): input stream
        filename (str): filename to write
        json (bool): load file line as json
    '''
    def __init__(self, node, filename='', json=True):
        async def _file(data):
            async with aiofiles.open(filename, mode='a') as fp:
                if json:
                    fp.write(JSON.dumps(data))
                else:
                    fp.write(data)
            return data

        super().__init__(foo=_file, name='File', inputs=1)
        node._downstream.append((self, 0))
        self._upstream.append(node)
