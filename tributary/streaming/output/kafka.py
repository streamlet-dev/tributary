import json as JSON
from confluent_kafka import Producer
from ..base import Node


class Kafka(Node):
    '''Connect to kafka server and send data

    Args:
        foo (callable): input stream
        foo_kwargs (dict): kwargs for the input stream
        servers (list): kafka bootstrap servers
        group (str): kafka group id
        topics (list): list of kafka topics to connect to
        json (bool): load input data as json
        wrap (bool): wrap result in a list
        interval (int): kafka poll interval
    '''

    def __init__(self, node, servers='', topic='', json=False, wrap=False):
        p = Producer({'bootstrap.servers': servers})

        async def _send(data, producer=p, topic=topic, json=json, wrap=wrap):
            ret = []
            # Trigger any available delivery report callbacks from previous produce() calls
            producer.poll(0)

            if wrap:
                data = [data]

            if json:
                data = JSON.dumps(data)

            producer.produce(topic, data.encode('utf-8'), callback=lambda *args: ret.append(args))

            for data in ret:
                yield data
            ret = []

        super().__init__(foo=_send, name='Kafka', inputs=1)
        node._downstream.append((self, 0))
        self._upstream.append(node)
