import os
import time
import tributary.streaming as ts


class TestFile:
    def setup(self):
        time.sleep(0.5)

    def test_file(self):
        file = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "test_file_data.json")
        )
        if os.path.exists(file):
            os.remove(file)

        def foo():
            yield 1
            yield 2
            yield 3
            yield 4

        def read_file(file):
            with open(file, "r") as fp:
                data = fp.read()
            return [int(x) for x in data]

        # Test that output is equal to what is read (generalized)
        out = ts.FileSink(ts.Foo(foo), filename=file, json=True)
        assert ts.run(out) == read_file(file)
