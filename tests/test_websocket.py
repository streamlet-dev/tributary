from mock import MagicMock, patch


class TestConfig:
    def setup(self):
        pass
        # setup() before each test method

    def teardown(self):
        pass
        # teardown() after each test method

    @classmethod
    def setup_class(cls):
        pass
        # setup_class() before any methods in this class

    @classmethod
    def teardown_class(cls):
        pass
        # teardown_class() after any methods in this class

    def test_grid(self):
        with patch('time.sleep') as m:
            from tributary.sources.websocket import WebSocketSource
            w = WebSocketSource('test')
            w.ws.on_error('test', 'test')
            w.ws.on_close('test')
            w.ws.on_open('test')
