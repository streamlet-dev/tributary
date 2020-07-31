import asyncpg
import asyncio
from .input import Foo


class Postgres(Foo):
    '''Connects to Postgres and yields result of query

        Args:
            user (str): postgres user
            password (str): postgres password
            database (str): postgres database
            host (str): postgres host
            queries (str): list of queries to execute
            interval (int): seconds to wait before executing queries
            repeat (int): times to repeat
    '''
    def __init__(self, user, password, database, host,
                 queries, repeat=1, interval=1):
        async def _send(queries=queries, repeat=int(repeat),
                        interval=int(interval), user=user,
                        password=password, database=database, host=host):
            print(user, password, database, host)
            conn = await asyncpg.connect(user=user, password=password,
                                         database=database, host=host.split(':')[0],
                                         port=host.split(':')[1])
            count = 0
            while count < repeat:
                data = []
                count += 1
                for query in queries:
                    values = await conn.fetch(query)
                    data.extend([list(x.items()) for x in values])
                yield data
                if interval:
                    await asyncio.sleep(interval)
            await conn.close()

        super().__init__(foo=_send)
        self._name = 'Postgres'
