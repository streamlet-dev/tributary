import ujson
from confluent_kafka import Producer
from ..base import _wrap


def Kafka(foo, foo_kwargs=None, **kafka_kwargs):
    return AsyncKafka(foo, foo_kwargs, **kafka_kwargs)


def AsyncKafka(foo, foo_kwargs=None, servers='', topic='', json=False, wrap=False):
    foo = _wrap(foo, foo_kwargs or {})

    p = Producer({'bootstrap.servers': servers})

    async def _send(foo, producer, topic, json, wrap):
        ret = []
        async for data in foo():
            # Trigger any available delivery report callbacks from previous produce() calls
            producer.poll(0)

            if wrap:
                data = [data]

            if json:
                data = ujson.dumps(data)

            producer.produce(topic, data.encode('utf-8'), callback=lambda *args: ret.append(args))

            for data in ret:
                yield data
            ret = []

    return _wrap(_send, dict(foo=foo, producer=p, topic=topic, json=json, wrap=wrap), name='Kafka')
