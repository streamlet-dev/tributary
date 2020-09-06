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
        self._consumer = None

        async def _listen(json=json, wrap=wrap, interval=interval):
            if self._consumer is None:

                if isinstance(topics, (list, tuple)):
                    topics = [topics]

                self._consumer = AIOKafkaConsumer(
                    *topics,
                    bootstrap_servers=servers,
                    group_id=group,
                    **consumer_kwargs)

                # Get cluster layout and join group `my-group`
                await self._consumer.start()

            async for msg in self._consumer:
                # Consume messages
                # msg.topic, msg.partition, msg.offset, msg.key, msg.value, msg.timestamp

                if json:
                    msg.value = JSON.loads(msg.value)
                if wrap:
                    msg.value = [msg.value]
                yield msg

            # Will leave consumer group; perform autocommit if enabled.
            await self._consumer.stop()

        super().__init__(foo=_listen)
        self._name = 'Kafka'
