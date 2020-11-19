import json as JSON
from socketIO_client_nexus import SocketIO as SIO
from urllib.parse import urlparse
from .output import Foo
from ..node import Node


class SocketIO(Foo):
    """Connect to socketIO server and send updates

    Args:
        node (Node): input stream
        url (str): url to connect to
        channel (str): socketio channel to connect through
        field (str): field to index result by
        sendinit (list): data to send on socketio connection open
        json (bool): load websocket data as json
        wrap (bool): wrap result in a list
        interval (int): socketio wai interval
    """

    def __init__(
        self,
        node,
        url,
        channel="",
        field="",
        sendinit=None,
        json=False,
        wrap=False,
        interval=1,
    ):
        o = urlparse(url)
        socketIO = SIO(o.scheme + "://" + o.netloc, o.port)

        if sendinit:
            socketIO.emit(sendinit)

        def _sio(data, field=field, json=json, wrap=wrap, interval=interval):
            if json:
                data = JSON.loads(data)

            if field:
                data = data[field]

            if wrap:
                data = [data]

            socketIO.emit(data)
            socketIO.wait(seconds=interval)
            return data

        super().__init__(foo=_sio, name="SocketIO", inputs=1)
        node >> self


Node.socketio = SocketIO
