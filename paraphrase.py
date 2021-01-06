import argparse
import time
import os
import ntpath
import git
import uuid
from paraphraser.paraphraser import Paraphraser
import json
import logging
from paraphraser.utils import init_logging, deterministic
import pandas as pd
import datetime


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Paraphraser')
    parser.add_argument('--method', default='roundtrip', choices=['roundtrip', 'intermediate', 'ncycles', 'dummy'])
    parser.add_argument('--translators', type=str, help='Translator names separated by whitespaces.', nargs='+',
                        default=[])
    parser.add_argument('--input', type=str, help='Either sentence to translate, between quotes (""), or'
                                                  'path to input (sentence by sentence in a .txt file).',
                        default=os.path.join('input', 'example.txt'))
    args = parser.parse_args()
    deterministic(42)

    t0 = datetime.datetime.now().timestamp()

    if len(args.input) < 4 or (len(args.input) > 4 and (args.input[-4:] not in ['.txt', '.csv'] or
                                                        (args.input[-4:] == '.txt' and len(args.input.split()) != 1))):
        paraphraser = Paraphraser.build(args.method, args.translators)
        for p in paraphraser.paraphrase(args.input):
            print(p)
        t1 = datetime.datetime.now().timestamp()
        print(f'Elapsed {t1 - t0}s')
        exit()

    timestamp = time.strftime("%Y-%m-%d-%H%M")
    output_path = 'output'
    input_name = ntpath.basename(args.input)
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    extra_id = uuid.uuid4().hex
    output_dir = os.path.join(output_path,
                              f'{args.method}-{"_".join(args.translators)}-{input_name}-{timestamp}-{sha[:4]}-'
                              f'{extra_id[:4]}')

    os.makedirs(output_dir)
    init_logging(os.path.join(output_dir, 'paraphrase.log'))
    paraphraser = Paraphraser.build(args.method, args.translators)
    if args.input[-4:] == '.csv':
        df = pd.read_csv(args.input)
        sentences = df['text'].values.tolist()
    else:
        with open(args.input, 'r') as f:
            sentences = f.readlines()
    paraphrases = paraphraser.paraphrase_sentences(sentences)
    with open(os.path.join(output_dir, 'paraphrases.json'), 'w') as f:
        json.dump(paraphrases, f, indent=4)

    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    t1 = datetime.datetime.now().timestamp()
    logging.info(f'Total: Elapsed {t1 - t0}s')

