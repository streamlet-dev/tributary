import asyncpg
from ..node import Node


def Postgres(node, user, password, database, host, query_parser):
    """Connects to Postgres and executes queries

    Args:
        node (Node): input tributary
        user (str): postgres user
        password (str): postgres password
        database (str): postgres database
        host (str): postgres host
        query_parser (func): parse input node data to query list
    """

    async def _send(
        data,
        query_parser=query_parser,
        user=user,
        password=password,
        database=database,
        host=host,
    ):
        conn = await asyncpg.connect(
            user=user,
            password=password,
            database=database,
            host=host.split(":")[0],
            port=host.split(":")[1],
        )
        queries = query_parser(data)
        for q in queries:
            await conn.execute(q)

        await conn.close()
        return data

    ret = Node(foo=_send, name="PostgresSink", inputs=1)
    node >> ret
    return ret
