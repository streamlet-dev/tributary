import tributary.streaming as ts


class TestPostgres:
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