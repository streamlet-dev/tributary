from json import loads as load_json
from websocket import create_connection
from ..base import StreamNone, StreamEnd
from ..thread import run


def ws(url, callback, json=False, wrap=False):
    ws = create_connection(url)
    for x in run(ws.recv):
        if isinstance(x, StreamNone):
            continue
        elif not x or isinstance(x, StreamEnd):
            break

        if json:
            x = load_json(x)
        if wrap:
            x = [x]
        callback(x)
