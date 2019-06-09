from ..base import _wrap  # noqa: F401


def File(filename, json=True):

    def _file(filename, json):
        with open(filename, 'r') as fp:
            for line in fp:
                if json:
                    yield json.loads(line)
                else:
                    yield line

    return _wrap(_file, dict(filename=filename, json=json), name='File')
