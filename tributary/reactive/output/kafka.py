import ujson
from confluent_kafka import Producer
from ..base import _wrap


def Kafka(foo, foo_kwargs=None, **kafka_kwargs):
    return SyncKafka(foo, foo_kwargs, **kafka_kwargs)


def SyncKafka(foo, foo_kwargs=None, servers='', topic='', json=False, wrap=False):
    foo_kwargs = foo_kwargs or {}
    foo = _wrap(foo, foo_kwargs)

    p = Producer({'bootstrap.servers': servers})

    def _send(foo, producer, topic, json, wrap):
        ret = []
        for data in foo():
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
