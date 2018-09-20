import requests
import time
from json import loads as load_json
from websocket import create_connection
from socketIO_client_nexus import SocketIO as SIO
try:
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse


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


def http(url, callback, interval=1, repeat=1, json=False, wrap=False, field=None, proxies=None, cookies=None):
    count = 0
    while count < repeat:
        msg = requests.get(url, cookies=cookies, proxies=proxies)

        if msg is None or msg.status_code != 200:
            break

        if json:
            msg = msg.json()

        if field:
            msg = msg[field]

        if wrap:
            msg = [msg]

        callback(msg)

        if interval:
            time.sleep(interval)
        if repeat >= 0:
            count += 1


def socketio(url, callback, channel='', field='', sendinit=None, json=False, wrap=False, interval=1):
    o = urlparse(url)
    socketIO = SIO(o.scheme + '://' + o.netloc, o.port)
    if sendinit:
        socketIO.emit(sendinit)

    while True:
        _data = []
        socketIO.on(channel, lambda data: _data.append(data))
        socketIO.wait(seconds=interval)
        for msg in _data:
            if json:
                msg = json.loads(msg)

            if field:
                msg = msg[field]

            if wrap:
                msg = [msg]

            callback(msg)
