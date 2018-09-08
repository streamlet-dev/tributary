import time
from IPython import get_ipython
from ..utils import queue_get_all, messages_to_json


class CommHandler(object):
    def __init__(self, q, channel, sleep=1, replay=True, replay_count=1000):
        self.closed = False
        self.q = q
        self.channel = channel
        self.target_name = 'lantern.live'
        self.opened = False
        self.sleep = sleep

        self.replay = replay
        self.replay_count = replay_count
        self.replay_log = []

        def on_close(msg):
            self.opened = False
            print('comm closed')

        def handle_open(comm, msg):
            print('comm open')
            self.opened = True
            comm.on_close = on_close
            self.comm = comm

        get_ipython().kernel.comm_manager.register_target(self.target_name + '/' + str(channel), handle_open)

    def run(self):
        # TODO wait until JS ready
        while not self.opened:
            time.sleep(self.sleep)

        while self.opened:
            messages = queue_get_all(self.q)
            if messages:
                # if self.replay:
                #     # TODO only the first `self.replay_count`
                #     self.replay_log.extend(messages)

                self.comm.send(data=messages_to_json(messages))
            time.sleep(self.sleep)


def runComm(q, channel, sleep=1):
    # print('adding handler %s%s' % ('lantern.live/', channel))
    comm = CommHandler(q, channel, sleep)
    comm.run()
