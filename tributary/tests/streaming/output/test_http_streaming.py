import tributary.streaming as ts
import pytest


class TestHttp:
    @pytest.mark.skipif("int(os.environ.get('TRIBUTARY_SKIP_DOCKER_TESTS', '1'))")
    def test_http(self):
        '''Test http server'''
        def foo():
            yield 'x'
            yield 'y'
            yield 'z'

        out = ts.HTTPSink(ts.Foo(foo), url='http://localhost:8080')
        assert len(ts.run(out)) == 3
