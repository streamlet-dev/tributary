import base64

import emails
from bs4 import BeautifulSoup

from ..node import Node
from .output import Foo


def make_email(html, from_, subject="", attachments=None):
    """Helper function to make emails

    html (str): content
    from_ (str): address to send the email from
    subject (str): subject of email
    attachments (list) : attachments to send
    """
    message = emails.html(charset="utf-8", subject=subject, html=html, mail_from=from_)
    soup = BeautifulSoup(html, "html.parser")

    # strip markdown links
    for item in soup.findAll("a", {"class": "anchor-link"}):
        item.decompose()

    # strip matplotlib base outs
    for item in soup.find_all(
        "div", class_="output_text output_subarea output_execute_result"
    ):
        for c in item.contents:
            if "&lt;matplotlib" in str(c):
                item.decompose()

    # remove dataframe table borders
    for item in soup.findAll("table", {"border": 1}):
        item["border"] = 0
        item["cellspacing"] = 0
        item["cellpadding"] = 0

    # extract imgs for outlook
    imgs = soup.find_all("img")
    imgs_to_attach = {}

    # attach main part
    for i, img in enumerate(imgs):
        if not img.get("localdata"):
            continue
        imgs_to_attach[img.get("cell_id") + "_" + str(i) + ".png"] = base64.b64decode(
            img.get("localdata")
        )
        img["src"] = "cid:" + img.get("cell_id") + "_" + str(i) + ".png"
        # encoders.encode_base64(part)
        del img["localdata"]

    # assemble email soup
    soup = str(soup)
    message = emails.html(charset="utf-8", subject=subject, html=soup, mail_from=from_)

    for img, data in imgs_to_attach.items():
        message.attach(filename=img, content_disposition="inline", data=data)

    return message


class Email(Foo):
    """Send an email

    Args:
        node (Node): input stream
        to (str): email address/es to send to
        smpt (dict): stmp settings for email account
    """

    def __init__(
        self,
        node,
        to,
        smtp,
    ):
        async def _send(
            message,
            to=to,
            smtp=smtp,
        ):
            r = message.send(to=to, smtp=smtp)
            return r, message

        super().__init__(foo=_send, inputs=1)
        self._name = "Email"
        node >> self


Node.email = Email
