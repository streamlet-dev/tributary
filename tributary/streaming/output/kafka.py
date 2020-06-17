import json as JSON
from confluent_kafka import Producer
from .output import _OUTPUT_GRAPHVIZSHAPE
from ..base import Node


def Kafka(node, servers='', topic='', json=False, wrap=False):
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

    p = Producer({'bootstrap.servers': servers})

    def _send(data, producer=p, topic=topic, json=json, wrap=wrap):
        # Trigger any available delivery report callbacks from previous produce() calls
        producer.poll(0)

        if wrap:
            data = [data]

        if json:
            data = JSON.dumps(data)

        producer.produce(topic, data.encode('utf-8'))
        return data

    ret = Node(foo=_send, name='Kafka', inputs=1, graphvizshape=_OUTPUT_GRAPHVIZSHAPE)
    node._downstream.append((ret, 0))
    ret._upstream.append(node)
    return ret
