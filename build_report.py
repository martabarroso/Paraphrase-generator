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
    table_intrinsic = {}
    table_extrinsic = {'test_accuracy': []}
    sample = {}
    for report_path in eval_report_paths:
        report = json.load(open(report_path))
        # name = os.path.dirname(report_path)
        name = os.path.basename(os.path.normpath(report_path))
        if name.startswith('baseline'):
            if 'name' not in table_extrinsic:
                table_extrinsic['name'] = [name]
            else:
                table_extrinsic['name'].append(name)
            table_extrinsic['test_accuracy'].append(report['test_accuracy'])
            continue
        if 'name' not in table_intrinsic:
            table_intrinsic['name'] = [name]
        else:
            table_intrinsic['name'].append(name)
        for key in report['IntrinsicEvaluator']['statistics']:
            mean = report['IntrinsicEvaluator']['statistics'][key]['mean']
            std = report['IntrinsicEvaluator']['statistics'][key]['std']
            value = f"{mean:.2f}+-{std:.2f}"
            if key not in table_intrinsic:
                table_intrinsic[key] = [value]
            else:
                table_intrinsic[key].append(value)
        np.random.shuffle(report['IntrinsicEvaluator']['results'])
        sample[name] = report['IntrinsicEvaluator']['results'][:5]
        if 'name' not in table_extrinsic:
            table_extrinsic['name'] = [name]
        else:
            table_extrinsic['name'].append(name)
        table_extrinsic['test_accuracy'].append(report['ExtrinsicEvaluator']['test_accuracy'])

    df = pd.DataFrame.from_dict(table_intrinsic)
    print('INTRINSIC DATAFRAME')
    print()
    print(df)
    print()
    print('INTRINSIC LATEX')
    print(df.to_latex())
    print()
    df = pd.DataFrame.from_dict(table_extrinsic)
    print('EXTRINSIC DATAFRAME')
    print()
    print(df)
    print()
    print('EXTRINSIC LATEX')
    print(df.to_latex())
    print()
    print("SAMPLE")
    pprint(sample)
