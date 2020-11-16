import tributary.streaming as ts
import requests
import time
import pytest


class TestHttp:
    def setup(self):
        time.sleep(0.5)

    @pytest.mark.skipif("int(os.environ.get('TRIBUTARY_SKIP_DOCKER_TESTS', '1'))")
    def test_http(self):
        """Test http server"""

        def foo():
            yield "x"
            yield "y"
            yield "z"

        out = ts.HTTPSink(ts.Foo(foo), url="http://localhost:8080")
        assert len(ts.run(out)) == 3

    def test_http_server(self):
        inp = ts.Random(interval=1, count=2)
        ss = ts.HTTPServerSink(inp, json=True, port=12345)
        w = ts.Window(ss)
        ts.run(w, blocking=False)

        time.sleep(1)
        resp = requests.get("http://127.0.0.1:12345/")
        print(resp.json())
        time.sleep(1)
        resp = requests.get("http://127.0.0.1:12345/")
        print(resp.json())
        time.sleep(2)
        print(w._accum)
        assert len(w._accum) == 2
