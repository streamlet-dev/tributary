
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
        import queue
        from tributary.utils import queue_get_all
        q = queue.Queue()
        q.put('test')
        assert queue_get_all(q) == ["test"]

    def test_messages_to_json(self):
        from tributary.utils import messages_to_json
        print(messages_to_json(['test']))
        assert messages_to_json(['test']) == '[test]'
        assert messages_to_json([{'test': 1}]) == '[{"test":1}]'
