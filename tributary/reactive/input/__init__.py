import time
import pandas as pd
from ..utils import Foo
from ..base import _wrap


def Random():

    def _random():
        step = 0
        while True:
            x = pd.util.testing.getTimeSeriesData()
            step += 1
            for i in range(len(x['A'])):
                yield {'A': x['A'][i],
                       'B': x['B'][i],
                       'C': x['C'][i],
                       'D': x['D'][i]}
                time.sleep(1)

    return _wrap(_random, {}, name='Random')
