import lantern as l
import datetime
import time
from builtins import range
from ..base import Streaming


class RandomSource(Streaming):
    def run(self):
        while True:
            df = l.ohlcv.sample().iloc[1:5]
            df['key'] = ['A', 'B', 'C', 'D']
            df.index = [df.index[0], df.index[0], df.index[0], df.index[0]]
            for i in range(len(df)):
                self.on_data(df.iloc[i].to_json())
            time.sleep(.1)


class RandomSource2(Streaming):
    def run(self):
        step = 0
        while True:
            df = l.ohlcv.sample().iloc[1:5]
            df['key'] = ['A', 'B', 'C', 'D']
            df.index = [df.index[0], df.index[0], df.index[0], df.index[0]]
            df.index += datetime.timedelta(days=step)
            df.index = df.index.astype(str)
            step += 1
            df = df.reset_index()
            for i in range(len(df)):
                self.on_data(df.iloc[i].to_json())
            time.sleep(1)
