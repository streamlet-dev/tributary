import sys
import time
from io import StringIO
from datetime import datetime, timedelta
from multiprocessing import Process
from threading import Thread
from .node import Node
from ..base import TributaryException


def _waitToRun(stdout, stderr, startsafter, endsafter, graph):
    # remap outputs
    # TODO
    # sys.stderr = stderr
    # sys.stdout = stdout

    # Fix issue with aioconsole
    Node.print._multiprocess = "yes"

    if (startsafter - datetime.now()).total_seconds() > 0:
        # in the future
        time.sleep((startsafter - datetime.now()).total_seconds())

    try:
        graph.run(blocking=True, newloop=True)
    except KeyboardInterrupt:
        sys.exit(0)
    except BaseException:
        sys.exit(1)
    else:
        # TODO do this?
        # time.sleep((endsafter - datetime.now()).total_seconds())
        sys.exit(0)


def _printOuts(stdout, stderr):
    if stdout:
        print(stdout, file=sys.stdout)
    if stderr:
        print(stderr, file=sys.stderr)


def _monitor(scheduler):
    while not scheduler.shouldend():
        # Monitor that process is running
        if not scheduler.alive():
            # restart subprocess if exit uncleanly
            if scheduler.exitcode() == 0:
                print("exiting...")
                return
            print("restarting...")
            scheduler.restart()

        # sleep for a second
        time.sleep(1)
        _printOuts(*scheduler.output())

    # dump last
    _printOuts(*scheduler.output())


class Scheduler(object):
    def __init__(self, graph, startsafter=None, endsafter=None):
        """Construct a new scheduler object

        Args:
            graph (StreamingGraph): the graph object to run
            end (Optional[datetime]): when to terminate the graph
        """
        # Graph object
        self._graph = graph

        # When to start and end
        self._startsafter = startsafter or datetime.now()
        self._endsafter = (
            self._startsafter + timedelta(days=1) if not endsafter else endsafter
        )

        # Runner process
        self._process = None

        # IO of subprocess
        self._stdout = None
        self._stderr = None

        # Monitor process
        self._monitor = Thread(target=_monitor, args=(self,))
        self._monitor.daemon = True

    def startsafter(self):
        return self._startsafter

    def endsafter(self):
        return self._endsafter

    def shouldend(self):
        return datetime.now() > self.endsafter()

    def alive(self):
        return self._process.is_alive()

    def running(self):
        # two conditions for running
        # 1. process is alive
        # 2. process exited uncleanly before end time
        return self.alive() or (
            self.exitcode() != 0 and datetime.now() < self.endsafter()
        )

    def exitcode(self):
        return self._process.exitcode

    def output(self):
        return self._stdout.getvalue(), self._stderr.getvalue()

    def restart(self):
        """restart the subprocess

        Note: this does not check the status of the subprocess
        """
        # IO of subprocess
        self._stdout = StringIO()
        self._stderr = StringIO()

        # assert no running process or that process is dead
        assert (self._process is None) or (not self._process.is_alive())

        # instantiate process
        self._process = Process(
            target=_waitToRun,
            args=(
                self._stdout,
                self._stderr,
                self._startsafter,
                self._endsafter,
                self._graph,
            ),
        )

        # Start subprocess
        self._process.start()

    def start(self):
        """Start running graph in a subprocess and monitor it.

        Notes: subprocess creates a copy of graph, but in this case we want
        that so that we have a "fresh" copy to restart in case of process
        dying"""
        self.restart()

        # Start monitor thread
        self._monitor.start()

    def stop(self, kill_immediately=False, interval=10):
        """Terminate the subprocess by first attempting to terminate it, then killing it.

        The default process is:
            SIGTERM
            wait `interval`
            SIGTERM
            wait `interval`
            SIGKILL

        Args:
            kill_immediately (bool): don't attempt to sigterm, just sigkill
            interval (int): time between sigterm, sigkill
        """
        if self._process is None:
            raise TributaryException("Process not yet started!")
        if not kill_immediately:
            print("terminating...")
            self._process.terminate()
            if self.running():
                print("terminating...")
                time.sleep(interval)
                self._process.terminate()
                if self.running():
                    time.sleep(interval)
        if self.running():
            print("killing...")
            self._process.kill()
        self._process.join()
