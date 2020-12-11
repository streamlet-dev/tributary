import asyncio
import time
import tributary.streaming as ts
import requests

def create(interval):
    async def foo_():
        for _ in range(5):
            yield interval
            await asyncio.sleep(interval)
    return foo_


fast = ts.Foo(create(1))
med = ts.Foo(create(2))
slow = ts.Foo(create(3))

def reducer(fast, med, slow):
    return {"fast": fast, "med": med, "slow": slow}

node = ts.Reduce(fast, med, slow, reducer=reducer).print()
ts.run(node, period=1)
