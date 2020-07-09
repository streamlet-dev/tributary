import aiofiles
import json as JSON
from .output import _OUTPUT_GRAPHVIZSHAPE
from ..node import Node


def File(node, filename='', json=True):
    '''Open up a file and write lines to the file

    Args:
        node (Node): input stream
        filename (str): filename to write
        json (bool): load file line as json
    '''

    async def _file(data):
        async with aiofiles.open(filename, mode='a') as fp:
            if json:
                await fp.write(JSON.dumps(data))
            else:
                await fp.write(data)
        return data

    ret = Node(foo=_file, name='File', inputs=1, graphvizshape=_OUTPUT_GRAPHVIZSHAPE)
    node >> ret
    return ret
