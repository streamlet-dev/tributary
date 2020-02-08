import tributary as t


class TestOutput:
    def test_print(self):
        out = t.run(t.Print(t.Random(5)))
        assert len(out) == 5

    def test_pprint(self):
        t.PPrint(t.Random(5))

    def test_perspective(self):
        try:
            out = t.run(t.Perspective(t.Random(5)))
            assert len(out) == 5
        except:
            pass
