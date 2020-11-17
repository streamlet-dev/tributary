import asyncio
import pytest
import requests
import time
import tributary.streaming as ts


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

    @pytest.mark.skipif('os.name == "nt"')
    def test_http_server(self):
        inp = ts.Random(interval=1, count=2)
        ss = ts.HTTPServerSink(inp, json=True, port=12346)
        w = ts.Window(ss)
        out = ts.run(w, blocking=False)

        try:
            time.sleep(1.5)
            resp = requests.get("http://127.0.0.1:12346/")
            print(resp.json())
            time.sleep(1)
            resp = requests.get("http://127.0.0.1:12346/")
            print(resp.json())
            time.sleep(2)
            print(w._accum)
            assert len(w._accum) == 2
        finally:
            asyncio.set_event_loop(asyncio.new_event_loop())
            try:
                out.stop()
            except RuntimeError:
                pass
