import tornado.websocket
import tornado.web
import tornado.ioloop
import time
from tributary.streaming.input import _gen


class DummyWebSocket(tornado.websocket.WebSocketHandler):
    def open(self):
        print("WebSocket opened")
        i = 0
        x = {y: _gen() for y in ('A', 'B', 'C', 'D')}
        try:
            while i < len(x['A']):
                self.write_message({'A': x['A'][i],
                                    'B': x['B'][i],
                                    'C': x['C'][i],
                                    'D': x['D'][i]})
                i += 1
                time.sleep(.1)
        finally:
            print("WebSocket closed")
            self.close()

    def on_message(self, message):
        self.write_message(u"You said: " + message)

    def on_close(self):
        print("WebSocket closed")


def main():
    app = tornado.web.Application([(r"/", DummyWebSocket)])
    app.listen(8899)
    print('listening on %d' % 8899)
    tornado.ioloop.IOLoop.current().start()


if __name__ == '__main__':
    main()
