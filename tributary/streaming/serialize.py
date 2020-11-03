class NodeSerializeMixin(object):
    def save(self):
        """return a serializeable structure representing this node's state"""
        import dill

        ret = {}
        ret["id"] = self._id
        ret["graphvizshape"] = self._graphvizshape
        # self._dd3g = None  # TODO
        ret["name"] = self._name_only  # use name sans id

        ret["input"] = [dill.dumps(_) for _ in self._input]
        ret["active"] = [dill.dumps(_) for _ in self._active]
        ret[
            "downstream"
        ] = []  # TODO think about this more [_.save() for _ in self._downstream]
        ret["upstream"] = [_.save() for _ in self._upstream]

        ret["foo"] = dill.dumps(self._foo)
        ret["foo_kwargs"] = dill.dumps(self._foo_kwargs)

        ret["delay_interval"] = self._delay_interval
        ret["execution_max"] = self._execution_max
        ret["execution_count"] = self._execution_count

        ret["last"] = dill.dumps(self._last)
        ret["finished"] = self._finished
        ret["use_dual"] = self._use_dual

        ret["drop"] = self._drop
        ret["replace"] = self._replace
        ret["repeat"] = self._repeat

        ret["attrs"] = self._initial_attrs
        return ret

    @staticmethod
    def restore(ret, **extra_attrs):
        import dill
        from .node import Node

        # self._dd3g = None  # TODO

        # constructor args
        foo = dill.loads(ret["foo"])
        foo_kwargs = dill.loads(ret["foo_kwargs"])
        name = ret["name"]
        inputs = len(ret["input"])
        drop = ret["drop"]
        replace = ret["replace"]
        repeat = ret["repeat"]
        graphvizshape = ret["graphvizshape"]
        delay_interval = ret["delay_interval"]
        execution_max = ret["execution_max"]
        use_dual = ret["use_dual"]

        # construct node
        n = Node(
            foo=foo,
            foo_kwargs=foo_kwargs,
            name=name,
            inputs=inputs,
            drop=drop,
            replace=replace,
            repeat=repeat,
            graphvizshape=graphvizshape,
            delay_interval=delay_interval,
            execution_max=execution_max,
            use_dual=use_dual,
        )

        # restore private attrs
        n._id = ret["id"]
        n._name = "{}#{}".format(name, n._id)
        n._input = [dill.loads(_) for _ in ret["input"]]
        n._active = [dill.loads(_) for _ in ret["active"]]
        # n._downstream = [] # TODO upstream don't get saved
        n._upstream = [Node.restore(_) for _ in ret["upstream"]]

        # restore node relationship
        for up_node in n._upstream:
            up_node >> n

        n._execution_count = ret["execution_count"]
        n._last = ret["last"]
        n._finished = ret["finished"]
        n._initial_attrs = ret["attrs"]
        for k, v in extra_attrs.items():
            setattr(n, k, v)
        return n


if __name__ == "__main__":
    # Test script
    import asyncio
    import tributary.streaming as ts
    import time

    async def foo():
        await asyncio.sleep(2)
        return 1

    o = ts.Foo(foo, count=3).print()

    g = ts.run(o, blocking=False)
    time.sleep(3)
    g.stop()

    ser = o.save()

    n = ts.Node.restore(ser)

    ts.run(n)
