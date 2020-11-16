import os
import os.path
import requests
import time
import tributary.streaming as ts
from aiohttp import web


class TestHttp:
    def setup(self):
        time.sleep(0.5)

    def test_http(self):
        out = ts.Print(ts.HTTPSource(url="https://google.com"))
        ret = ts.run(out)
        print(ret)
        assert len(ret) == 1

    def test_http_server(self):
        ss = ts.HTTPServerSource(json=True, host="127.0.0.1", port=12345)
        w = ts.Window(ss)
        l = ts.run(w, blocking=False)

        resp = requests.post("http://127.0.0.1:12345/", json={"test": 1, "test2": 2})
        l.stop()
        assert w._accum == [{"test": 1, "test2": 2}]
