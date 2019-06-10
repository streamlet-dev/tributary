import time
import tributary.functional as t


class TestInput:
    def test_http(self):
        t.pipeline([t.http], ['callback'], [{'url': 'https://www.google.com'}], on_data=print)
        time.sleep(2)
        t.stop()
