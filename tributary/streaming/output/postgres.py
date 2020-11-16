import asyncpg
from .output import Foo
from ..node import Node


class Postgres(Foo):
    """Connects to Postgres and executes queries

    Args:
        node (Node): input tributary
        user (str): postgres user
        password (str): postgres password
        database (str): postgres database
        host (str): postgres host
        query_parser (func): parse input node data to query list
    """

    def __init__(self, node, user, password, database, host, query_parser):
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

        super().__init__(foo=_send, name="PostgresSink", inputs=1)
        node >> self


Node.postgres = Postgres
