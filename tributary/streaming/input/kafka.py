import json as JSON
from aiokafka import AIOKafkaConsumer
from .input import Foo


class Kafka(Foo):
    '''Connect to kafka server and yield back results

    Args:
        servers (list): kafka bootstrap servers
        group (str): kafka group id
        topics (list): list of kafka topics to connect to
        json (bool): load input data as json
        wrap (bool): wrap result in a list
        interval (int): kafka poll interval
    '''

    def __init__(self, servers, group, topics, json=False, wrap=False, interval=1, **consumer_kwargs):
        consumer = AIOKafkaConsumer(
            *topics,
            bootstrap_servers=servers,
            group_id=group,
            **consumer_kwargs)

        async def _listen(consumer=consumer, json=json, wrap=wrap, interval=interval):
            # Get cluster layout and join group `my-group`
            await consumer.start()

            async for msg in consumer:
                # Consume messages
                # msg.topic, msg.partition, msg.offset, msg.key, msg.value, msg.timestamp

                if json:
                    msg.value = JSON.loads(msg.value)
                if wrap:
                    msg.value = [msg.value]
                yield msg

            # Will leave consumer group; perform autocommit if enabled.
            await consumer.stop()

        super().__init__(foo=_listen)
        self._name = 'Kafka'
