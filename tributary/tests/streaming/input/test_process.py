import os.path
import pytest
import time

import tributary.streaming as ts

vals = [
    "__init__.py",
    "__pycache__",
    "test_file_data.csv",
    "test_file_data.json",
    "test_file_streaming.py",
    "test_http_streaming.py",
    "test_input_streaming.py",
    "test_postgres_streaming.py",
    "test_process.py",
    "test_sse_streaming.py",
]


class TestProcess:
    def setup(self):
        time.sleep(0.5)

    @pytest.mark.skipif("'--cov=tributary' in sys.argv")
    def test_process(self):
        path = os.path.dirname(__file__)
        ret = ts.run(ts.SubprocessSource("ls {}".format(path)))
        print(ret)
        assert ret == vals

    @pytest.mark.skipif("'--cov=tributary' in sys.argv")
    def test_process_one_off(self):
        path = os.path.dirname(__file__)
        ret = ts.run(ts.SubprocessSource("ls {}".format(path), one_off=True))
        print(ret)
        assert ret == ["\n".join(vals) + "\n"]
