import tributary.streaming as ts
import pytest
import time


class TestPostgres:
    def setup(self):
        time.sleep(0.5)

    @pytest.mark.skipif("int(os.environ.get('TRIBUTARY_SKIP_DOCKER_TESTS', '1'))")
    def test_pg(self):
        def foo():
            yield 1
            yield 2
            yield 3

        def parser(data):
            return ["INSERT INTO test(col1) VALUES ({});".format(data)]

        out = ts.PostgresSink(
            ts.Foo(foo),
            query_parser=parser,
            user="postgres",
            database="postgres",
            password="test",
            host="localhost:5432",
        )
        assert len(ts.run(out)) == 3
