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
        with patch('tributary.hosts.comm.get_ipython') as m, \
             patch('time.sleep') as m2, \
             patch('tributary.hosts.comm.queue_get_all') as m3:

            m.return_value = MagicMock()
            m.return_value.kernel = MagicMock()
            m.return_value.kernel.comm_manager = MagicMock()
            m.return_value.kernel.comm_manager.register_target = MagicMock()

            from tributary.hosts.comm import CommHandler
            c = CommHandler(MagicMock(), MagicMock())
            c.comm = MagicMock()

            def foo(arg):
                c.opened = not c.opened
            m2.side_effect = foo

            c.run()
