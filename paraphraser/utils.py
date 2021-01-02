import logging
import re

# https://gist.github.com/uogbuji/705383
GRUBER_URLINTEXT_PAT = re.compile(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019]))')


def init_logging(name: str):
    logging.basicConfig(filename=name, level=logging.INFO)
    logging.getLogger('').addHandler(logging.StreamHandler())


def normalize_spaces_remove_urls(s: str) -> str:
    s = GRUBER_URLINTEXT_PAT.sub('', s)
    return ' '.join(s.split())
