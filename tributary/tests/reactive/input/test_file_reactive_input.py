import tributary as t
import os
import os.path


class TestFile:
    def test_file(self):
        file = os.path.join(os.path.dirname(__file__),
                            '..', '..', 'test_data',
                            'ohlc.csv')
        out = t.run(t.File(file, json=False))
        assert len(out) == 50
