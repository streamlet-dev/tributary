import tributary.streaming as ts
import pytest
import time


class TestWebSocket:
    def setup(self):
        time.sleep(0.5)

    @pytest.mark.skipif("int(os.environ.get('TRIBUTARY_SKIP_DOCKER_TESTS', '1'))")
    def test_websocket(self):
        """Test websocket streaming"""

        def foo():
            yield "x"
            yield "y"
            yield "z"

        out = ts.WebSocketSink(ts.Foo(foo), url="ws://localhost:8080", response=True)
        assert len(ts.run(out)) == 3

    def test_websocket_server(self):
        """Test websocket server"""

        def foo():
            yield "x"
            yield "y"
            yield "z"

        out = ts.WebSocketServerSink(ts.Foo(foo), port=1234)
        assert len(ts.run(out)) == 3
