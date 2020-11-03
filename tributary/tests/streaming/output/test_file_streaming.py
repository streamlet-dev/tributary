import os
import tributary.streaming as ts
import pytest


class TestFile:
    def test_file(self):
        file = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_file_data.csv'))
        if os.path.exists(file):
            os.remove(file)

        def foo():
            yield 1
            yield 2
            yield 3
            yield 4

        def read_file(file):
            with open(file, 'r') as fp:
                data = fp.read()
            return [int(x) for x in data]

        # Test that output is equal to what is read (generalized)
        out = ts.FileSink(ts.Foo(foo), filename=file)
        assert ts.run(out) == read_file(file)
