import tributary.streaming as ts
import pytest


class TestPostgres:
    @pytest.mark.skipif("int(os.environ.get('TRIBUTARY_SKIP_DOCKER_TESTS'))")
    def test_http(self):

        def foo():
            yield 1
            yield 2
            yield 3

        def parser(data):
            return ["INSERT INTO test(col1) VALUES ({});".format(data)]

        query = ['SELECT * FROM test']
        out = ts.PostgresSink(ts.Foo(foo),
                              query_parser=parser,
                              user='postgres',
                              database='postgres',
                              password='test',
                              host='localhost:5432')
        assert len(ts.run(out)) == 3
