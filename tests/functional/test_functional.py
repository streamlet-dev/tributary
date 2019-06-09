import time
import random
from perspective import PerspectiveWidget


class TestFunctional:
    def setup(self):
        pass
        # setup() before each test method

    def test_1(self):
        import tributary as t

        def foo1(on_data):
            x = 0
            while x < 100:
                on_data({'a': random.random(), 'b': random.randint(0, 1000), 'x': x})
                time.sleep(.1)
                x = x+1

        def foo2(data, callback):
            callback([{'a': data['a'] * 1000, 'b': data['b'], 'c': 'AAPL', 'x': data['x']}])

        p = PerspectiveWidget([], view='y_line', columns=['a', 'b'], rowpivots=['x'], colpivots=['c'])

        t.pipeline([foo1, foo2], ['on_data', 'callback'], on_data=p.update)
        time.sleep(10)
        t.stop()
