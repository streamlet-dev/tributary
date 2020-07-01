import tributary.streaming as ts


class TestKafka:
    def test_kafka(self):
        def foo():
            yield 'a'
            yield 'b'
            yield 'c'

        out = ts.KafkaSink(ts.Foo(foo), servers='localhost:9092', topic='tributary')
        assert ts.run(out) == ['a', 'b', 'c']
