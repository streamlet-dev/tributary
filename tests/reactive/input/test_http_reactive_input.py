

class TestFile:
    def test_file(self):
        import tributary as t
        out = t.run(t.HTTP('http://tim.paine.nyc', json=False))
        assert len(out) > 0
