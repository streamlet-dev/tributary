import aiofiles
import json as JSON
from .output import Foo
from ..node import Node


class File(Foo):
    """Open up a file and write lines to the file

    Args:
        node (Node): input stream
        filename (str): filename to write
        json (bool): write file line as json
        csv (bool): write file line as csv
    """

    def __init__(self, node, filename="", json=False, csv=False):
        async def _file(data):
            if csv:
                async with aiofiles.open(filename, "w") as f:
                    await f.write(",".join(data))
            else:
                async with aiofiles.open(filename, mode="a") as f:
                    if json:
                        await f.write(JSON.dumps(data))
                    else:
                        await f.write(data)
                return data

        super().__init__(foo=_file, name="File", inputs=1)
        node >> self


Node.file = File
