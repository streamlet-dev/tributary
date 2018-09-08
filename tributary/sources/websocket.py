import websocket
from ..base import Streaming


class WebSocketSource(Streaming):
    def __init__(self, addr, on_open=None, on_close=None, on_error=None):
        def on_message(ws, message):
            self.on_data(message)

        def on_error_default(ws, error):
            print(error)

        def on_close_default(ws):
            print("### closed ###")

        def on_open_default(ws):
            pass

        self.ws = websocket.WebSocketApp(addr,
                                         on_message=on_message,
                                         on_error=on_error if on_error else on_error_default,
                                         on_close=on_close if on_close else on_close_default)
        self.ws.on_open = on_open if on_open else on_open_default

    def run(self):
        self.ws.run_forever()
