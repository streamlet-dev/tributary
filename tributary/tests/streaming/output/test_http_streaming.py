import tributary.streaming as ts


class TestHttp:
    def test_http(self):
        '''Test http server'''
        def foo():
            yield 'x'
            yield 'y'
            yield 'z'

        out = ts.HTTPSink(ts.Foo(foo), url='http://localhost:8080')
        assert len(ts.run(out)) == 3
