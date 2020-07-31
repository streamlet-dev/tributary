import tributary.streaming as ts


class TestPostgres:
    def test_http(self):
        query = ['SELECT * FROM test']
        out = ts.PostgresSource(queries=query,
                          user='postgres',
                          database='postgres',
                          password='test',
                          host='localhost:5432')
        assert len(ts.run(out)) != 0