from paraphraser.evaluator import Evaluator
import argparse
import os
import json
import time
import git
import uuid
from paraphraser.utils import init_logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Paraphrases evaluator')
    parser.add_argument('output_path', help='Path to output paraphrases to evaluate',
                        default=os.path.join('output', 'example'))
    args = parser.parse_args()
    output = args.output_path
    timestamp = time.strftime("%Y-%m-%d-%H%M")
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    extra_id = uuid.uuid4().hex
    name = f'eval-{timestamp}-{sha[:4]}-{extra_id[:4]}'
    init_logging(os.path.join(args.output_path, name + '.log'))
    evaluators = Evaluator.get_all_evaluators()
    with open(os.path.join(output, 'paraphrases.json'), 'r') as f:
        sentences2paraphrases_dict = json.load(f)
    results = {}
    for evaluator in evaluators:
        results[evaluator.__class__.__name__] = evaluator.evaluate_paraphrases(sentences2paraphrases_dict)
    timestamp = time.strftime("%Y-%m-%d-%H%M")
    with open(os.path.join(args.output_path, name + '.json'), 'w') as f:
        json.dump(results, f, indent=4)
