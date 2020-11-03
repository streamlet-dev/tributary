import json as JSON
import sys
import os.path

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
)


if __name__ == "__main__":
    import tributary.streaming as ts

    def _json(val):
        return JSON.dumps(val)

    ts.run(ts.Console(json=True).apply(_json).print())
