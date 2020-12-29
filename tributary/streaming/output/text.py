from twilio.rest import Client

from ..node import Node
from .output import Foo


class TextMessage(Foo):
    """Send a text message

    Args:
        node (Node): input stream
        to (str): phone number/s to send to
        from_ (str): phone number to send from
        twilio (dict): twilio account info kwargs for twilio.rest.Client
    """

    def __init__(
        self,
        node,
        to,
        from_,
        twilio,
    ):
        self._twilio_client = Client(**twilio)

        async def _send(
            message,
            to=to,
            from_=from_,
            twilio=twilio,
        ):

            r = self._twilio_client.messages.create(to=to, from_=from_, body=message)

            return r, message

        super().__init__(foo=_send, inputs=1)
        self._name = "TextMessage"
        node >> self


Node.textMessage = TextMessage
