import tributary.streaming as ts
import pytest
import time


class TestKafka:
    def setup(self):
        time.sleep(0.5)

    @pytest.mark.skipif("int(os.environ.get('TRIBUTARY_SKIP_DOCKER_TESTS', '1'))")
    def test_kafka(self):
        """Test streaming with Kafka"""

        def foo():
            yield "a"
            yield "b"
            yield "c"

        out = ts.KafkaSink(ts.Foo(foo), servers="localhost:9092", topic="tributary")
        assert ts.run(out) == ["a", "b", "c"]
