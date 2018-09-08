import os
import os.path
from IPython import get_ipython
import queue
import threading
import ujson


def _q_me():
    q = queue.Queue()

    def qput(message):
        q.put(message)
    return q, qput


def _get_session():
    # TODO add secret
    p = os.path.abspath(get_ipython().kernel.session.config['IPKernelApp']['connection_file'])
    sessionid = p.split(os.sep)[-1].replace('kernel-', '').replace('.json', '')
    return sessionid


def _start_sender_thread(target, q, rank, sleep):
    t1 = threading.Thread(target=target, args=(q, str(rank), sleep))
    t1.start()
    return rank + 1


def queue_get_all(q):
    items = []
    while True:
        try:
            items.append(q.get_nowait())
        except queue.Empty:
            break
    return items


def messages_to_json(lst):
    if lst and isinstance(lst[0], str):
        # already jsons:
        return '[' + ','.join(lst) + ']'
    return ujson.dumps(lst)
