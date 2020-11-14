import logging


def init_logging(name: str):
    logging.basicConfig(filename=name, level=logging.INFO)
    logging.getLogger('').addHandler(logging.StreamHandler())
