import json as JSON
from aiokafka import AIOKafkaProducer
from .output import Foo
from ..node import Node


class Kafka(Foo):
    """Connect to kafka server and send data

    Args:
        node (Node): input tributary
        servers (list): kafka bootstrap servers
        topic (str): kafka topic to connect to
        json (bool): load input data as json
        wrap (bool): wrap result in a list
        interval (int): kafka poll interval
    """

    def __init__(
        self, node, servers="", topic="", json=False, wrap=False, **producer_kwargs
    ):
        async def _send(data, topic=topic, json=json, wrap=wrap):
            if self._producer is None:
                self._producer = AIOKafkaProducer(
                    bootstrap_servers=servers, **producer_kwargs
                )

                # Get cluster layout and initial topic/partition leadership information
                await self._producer.start()

            if wrap:
                data = [data]

            if json:
                data = JSON.dumps(data)

            # Produce message
            await self._producer.send_and_wait(topic, data.encode("utf-8"))
            return data

        # # Wait for all pending messages to be delivered or expire.
        # await producer.stop()
        super().__init__(foo=_send, name="Kafka", inputs=1)
        node >> self
        self.set("_producer", None)


Node.kafka = Kafka
