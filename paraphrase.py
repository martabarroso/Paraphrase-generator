import argparse
import time
import os
import ntpath
import git
import uuid
from paraphraser.paraphraser import Paraphraser
import json
from paraphraser.utils import init_logging


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Paraphraser')
    parser.add_argument('--method', default='roundtrip', choices=['roundtrip', 'intermediate', '2cycles', 'dummy'])
    parser.add_argument('--translators', type=str, help='Translator names separated by whitespaces.', nargs='+',
                        default=[])
    parser.add_argument('--input', type=str, help='Either sentence to translate, between quotes (""), or'
                                                  'path to input (sentence by sentence in a .txt file).',
                        default=os.path.join('input', 'example.txt'))
    parser.add_argument('--n', type=int, help='Number of generated paraphrases per sentence.', default=5)
    args = parser.parse_args()
    if len(args.input) < 4 or (len(args.input) > 4 and (args.input[-4:] != '.txt' or (args.input[-4:] == '.txt' and
                                                                                      len(args.input.split()) != 1))):
        paraphraser = Paraphraser.build(args.method, args.translators)
        for p in paraphraser.paraphrase(args.input, n_paraphrases=args.n):
            print(p)
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
    with open(args.input, 'r') as f:
        sentences = f.readlines()
    paraphrases = paraphraser.paraphrase_sentences(sentences, n_paraphrases_per_sentence=args.n)
    with open(os.path.join(output_dir, 'paraphrases.json'), 'w') as f:
        json.dump(paraphrases, f, indent=4)

    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)