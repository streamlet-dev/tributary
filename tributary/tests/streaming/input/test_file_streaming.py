import os
import os.path
import tributary.streaming as ts


class TestFile:
    def test_file(self):
        file = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_file_data.csv'))

        out = ts.Print(ts.File(filename=file))
        assert ts.run(out) == [1, 2, 3, 4]
