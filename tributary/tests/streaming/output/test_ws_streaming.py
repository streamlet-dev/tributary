import tributary.streaming as ts


class TestWebSocket:
    def test_websocket(self):
        def foo():
            yield 'x'
            yield 'y'
            yield 'z'

        out = ts.WebSocketSink(ts.Foo(foo), url='ws://localhost:8080', response=True)
        assert len(ts.run(out)) == 3

