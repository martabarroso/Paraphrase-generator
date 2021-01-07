import pandas as pd
import json
import sys
from paraphraser.utils import deterministic
import os
from pprint import pprint
import numpy as np

SEED = 42
SAMPLE = 5

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python build_report.py [PATH_EVAL_LOG_1] [PATH_EVAL_LOG_2] ...')
        exit()
    deterministic(SEED)
    eval_report_paths = sys.argv[1:]
    table = {}
    sample = {}
    for report_path in eval_report_paths:
        report = json.load(open(report_path))
        # name = os.path.dirname(report_path)
        name = os.path.basename(os.path.normpath(report_path))
        if 'name' not in table:
            table['name'] = [name]
        else:
            table['name'].append(name)
        for key in report['IntrinsicEvaluator']['statistics']:
            mean = report['IntrinsicEvaluator']['statistics'][key]['mean']
            std = report['IntrinsicEvaluator']['statistics'][key]['std']
            value = f"{mean:.2f}+-{std:.2f}"
            if key not in table:
                table[key] = [value]
            else:
                table[key].append(value)
        np.random.shuffle(report['IntrinsicEvaluator']['results'])
        sample[name] = report['IntrinsicEvaluator']['results'][:5]

    df = pd.DataFrame.from_dict(table)
    print('DATAFRAME')
    print()
    print(df)
    print()
    print('LATEX')
    print(df.to_latex())
    print()
    print("SAMPLE")
    pprint(sample)
