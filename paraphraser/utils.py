import logging
import re
import os
import torch
import numpy as np
import random

# https://gist.github.com/uogbuji/705383
GRUBER_URLINTEXT_PAT = re.compile(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019]))')


def init_logging(name: str):
    logging.basicConfig(filename=name, level=logging.INFO)
    logging.getLogger('').addHandler(logging.StreamHandler())


def normalize_spaces_remove_urls(s: str) -> str:
    s = GRUBER_URLINTEXT_PAT.sub('', s)
    return ' '.join(s.split())


def deterministic(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True