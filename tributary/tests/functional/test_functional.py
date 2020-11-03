import pytest
import random
import time


class TestFunctional:
    def setup(self):
        time.sleep(1)

    @pytest.mark.skipif("os.name == 'nt'")
    def test_general(self):
        import tributary.functional as t

        def foo1(on_data):
            x = 0
            while x < 5:
                on_data({"a": random.random(), "b": random.randint(0, 1000), "x": x})
                time.sleep(0.01)
                x = x + 1

        def foo2(data, callback):
            callback(
                [{"a": data["a"] * 1000, "b": data["b"], "c": "AAPL", "x": data["x"]}]
            )

        t.pipeline([foo1, foo2], ["on_data", "callback"], on_data=lambda x: None)
        time.sleep(1)
        t.stop()
