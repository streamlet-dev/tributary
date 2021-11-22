import asyncio

_DD3_TRANSITION_DELAY = 0.1  # used so you can visually see the
# transition e.g. not too fast


class _DagreD3Mixin(object):
    # ***********************
    # Dagre D3 integration
    # ***********************

    async def _startdd3g(self):
        """represent start of calculation with a dd3 node"""
        if self._dd3g:  # disable if not installed/enabled as it incurs a delay
            self._dd3g.setNode(self._name, tooltip=str(self._last), style="fill: #0f0")
            await asyncio.sleep(_DD3_TRANSITION_DELAY)

    async def _waitdd3g(self):
        """represent a node waiting for its input to tick"""
        if self._dd3g:  # disable if not installed/enabled as it incurs a delay
            self._dd3g.setNode(self._name, tooltip=str(self._last), style="fill: #ff0")
            await asyncio.sleep(_DD3_TRANSITION_DELAY)

    async def _finishdd3g(self):
        """represent a node that has finished its calculation"""
        if self._dd3g:  # disable if not installed/enabled as it incurs a delay
            self._dd3g.setNode(self._name, tooltip=str(self._last), style="fill: #f00")
            await asyncio.sleep(_DD3_TRANSITION_DELAY)

    async def _enddd3g(self):
        """represent a node that has finished all calculations"""
        if self._dd3g:  # disable if not installed/enabled as it incurs a delay
            self._dd3g.setNode(self._name, tooltip=str(self._last), style="fill: #fff")
            await asyncio.sleep(_DD3_TRANSITION_DELAY)

    # ***********************
