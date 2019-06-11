import tributary.asynchronous as t


class TestInput:
    def test_file(self):
        out = t.run(t.Random(5))
        assert len(out) == 5
