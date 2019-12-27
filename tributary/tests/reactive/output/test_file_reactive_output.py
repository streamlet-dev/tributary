import tempfile
import tributary as t


class TestFile:
    def test_file(self):
        with tempfile.NamedTemporaryFile() as f:
            out = t.run(t.FileSink(t.Const('test'), filename=f.name, json=False))
        assert len(out) == 1
