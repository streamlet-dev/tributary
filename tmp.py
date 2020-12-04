import time
import asyncio
import tributary.streaming as ts
from datetime import datetime, timedelta


async def fooFast():
    await asyncio.sleep(1)
    yield 1
    await asyncio.sleep(1)
    yield 1
    await asyncio.sleep(1)
    yield 1
    await asyncio.sleep(1)

async def fooMed():
    await asyncio.sleep(2)
    yield 2
    await asyncio.sleep(2)
    yield 2
    await asyncio.sleep(2)
    yield 2
    await asyncio.sleep(2)

async def fooSlow():
    await asyncio.sleep(4)
    yield 4
    await asyncio.sleep(4)
    yield 4
    await asyncio.sleep(4)
    yield 4
    await asyncio.sleep(4)


def reducer(one, two, three):
    return {"1": one, "2": two, "3": three}

n = ts.Reduce(ts.Foo(fooFast), ts.Foo(fooMed), ts.Foo(fooSlow), reducer=reducer).print()
ts.run(n)
