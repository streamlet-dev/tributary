import asyncio
from .input import Foo


class SubprocessSource(Foo):
    """Open up a subprocess and yield the results as they come

    Args:
        command (str): command to run
    """

    def __init__(self, command, one_off=False):
        async def _proc(command=command, one_off=one_off):
            proc = await asyncio.create_subprocess_shell(
                command, stdout=asyncio.subprocess.PIPE
            )

            if one_off:
                stdout, _ = await proc.communicate()

                if stdout:
                    stdout = stdout.decode()
                yield stdout

            else:
                while proc.returncode is None:
                    done = False

                    while not done:
                        val = await asyncio.create_task(proc.stdout.readline())
                        val = val.decode().strip()

                        if val == "":
                            done = True
                            break

                        yield val

        super().__init__(foo=_proc)
        self._name = "Proc"
