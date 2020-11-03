from tributary import TributaryException

from datetime import datetime, timedelta
from multiprocessing import Process


class Scheduler(object):
    def __init__(self, graph, startsafter=None, endsafter=None):
        '''Construct a new scheduler object

        Args:
            graph (StreamingGraph): the graph object to run
            end (Optional[datetime]): when to terminate the graph
        '''
        self._graph = graph
        self._startsafter = startsafter or datetime.now()
        self._endsafter = self._startsafter + timedelta(days=1) if not endsafter else endsafter
        self._process = None

    def start(self):
        '''Start running graph in a subprocess and monitor it.

        Notes: subprocess creates a copy of graph, but in this case we want
        that so that we have a "fresh" copy to restart in case of process
        dying'''

    def stop(self, kill_immediately=False, interval=10):
        '''Terminate the subprocess by first attempting to terminate it, then killing it.

        The default process is:
            SIGTERM
            wait `interval`
            SIGTERM
            wait `interval`
            SIGKILL
        
        Args:
            kill_immediately (bool): don't attempt to sigterm, just sigkill
            interval (int): time between sigterm, sigkill
        '''
        if self._process is None:
            raise TributaryException('Process not yet started!')


