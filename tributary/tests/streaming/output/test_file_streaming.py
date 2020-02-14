import os
import os.path
import tributary.streaming as ts


class TestFile:
    def test_file(self):
        file = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_file_data.csv'))

        def foo():
            yield 1
            yield 2
            yield 3
            yield 4

        out = ts.FileSink(ts.Foo(foo), filename=file)
        assert ts.run(out) == [1, 2, 3, 4]
