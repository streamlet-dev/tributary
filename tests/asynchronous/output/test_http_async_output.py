import tributary.asynchronous as t


class TestFile:
    def test_file(self):
        def foo():
            return 'test'
        out = t.run(t.HTTPSink(foo, url='http://tim.paine.nyc', json=False))
        assert len(out) > 0
