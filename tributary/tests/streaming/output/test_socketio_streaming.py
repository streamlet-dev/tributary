import tributary.streaming as ts
import pytest
import time


class TestSocketIO:
    def setup(self):
        time.sleep(0.5)

    @pytest.mark.skipif("int(os.environ.get('TRIBUTARY_SKIP_DOCKER_TESTS', '1'))")
    def test_socketio(self):
        """Test socketio streaming"""

        def foo():
            yield "a"
            yield "b"
            yield "c"

        out = ts.SocketIOSink(ts.Foo(foo), url="http://localhost:8069")
        assert ts.run(out) == ["a", "b", "c"]
