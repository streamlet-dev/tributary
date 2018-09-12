import time
import pandas as pd
from ..base import _wrap


def Random(size=10, interval=0.1):

    def _random(size, interval):
        step = 0
        while step < size:
            x = pd.util.testing.getTimeSeriesData()
            for i in range(len(x['A'])):
                if step >= size:
                    break
                yield {'A': x['A'][i],
                       'B': x['B'][i],
                       'C': x['C'][i],
                       'D': x['D'][i]}
                time.sleep(interval)
                step += 1

    return _wrap(_random, dict(size=size, interval=interval), name='Random')
