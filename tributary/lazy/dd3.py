class _DagreD3Mixin(object):
    def _greendd3g(self):
        if self._dd3g:
            self._dd3g.setNode(
                self._name, tooltip=str(self.value()), style="fill: #0f0"
            )

    def _yellowdd3g(self):
        if self._dd3g:
            self._dd3g.setNode(
                self._name, tooltip=str(self.value()), style="fill: #ff0"
            )

    def _reddd3g(self):
        if self._dd3g:
            self._dd3g.setNode(
                self._name, tooltip=str(self.value()), style="fill: #f00"
            )

    def _whited3g(self):
        if self._dd3g:
            self._dd3g.setNode(
                self._name, tooltip=str(self.value()), style="fill: #fff"
            )
