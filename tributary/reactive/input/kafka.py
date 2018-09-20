import ujson
from confluent_kafka import Consumer, KafkaError
from ..base import _wrap


def Kafka(servers, group, topics, json=False, wrap=False, interval=1):
    return SyncKafka(servers, group, topics, json=json, wrap=wrap, interval=interval)


def SyncKafka(servers, group, topics, json=False, wrap=False, interval=1):
    c = Consumer({
        'bootstrap.servers': servers,
        'group.id': group,
        'default.topic.config': {
            'auto.offset.reset': 'smallest'
        }
    })

    if not isinstance(topics, list):
        topics = [topics]
    c.subscribe(topics)

    def _listen(consumer, json, wrap, interval):
        while True:
            msg = consumer.poll(interval)

            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    print(msg.error())
                    break

            msg = msg.value().decode('utf-8')

            if not msg:
                break
            if json:
                msg = ujson.loads(msg)
            if wrap:
                msg = [msg]
            yield msg

    return _wrap(_listen, dict(consumer=c, json=json, wrap=wrap, interval=interval), name='Kafka')
