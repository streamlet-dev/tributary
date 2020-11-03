import os
import os.path
import tributary.streaming as ts


class TestHttp:
    def test_http(self):
        out = ts.Print(ts.HTTPSource(url="https://google.com"))
        ret = ts.run(out)
        print(ret)
        assert len(ret) == 1
