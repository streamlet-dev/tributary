import tributary.streaming as ts


class TestKafka:
    def test_kafka(self):
	'''Test streaming with Kafka'''
        def foo():
            yield 'a'
            yield 'b'
            yield 'c'

        out = ts.KafkaSink(ts.Foo(foo), servers='localhost:9092', topic='tributary')
        assert ts.run(out) == ['a', 'b', 'c']

