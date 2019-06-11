import tributary.asynchronous as t


class TestFile:
    def test_file(self):
        out = t.run(t.HTTP('https://bonds.paine.nyc', json=True))
        assert len(out) > 0
