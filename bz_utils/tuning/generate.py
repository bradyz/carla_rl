"""
Usage:

python3 -m bz_utils.tuning.generate --json example.json --output_dir ~/debug --splits 1

example.json:
{
    "setup": [
        "echo 123",
        "echo 456"
    ],
    "command": "python3 train_birdview.py",
    "fixed": {
        "dataset_dir": "/mnt/c/Users/Brady/Documents/code/bz_utils/tuning",
        "log_dir": "/mnt/c/Users/Brady/Documents/code/bz_utils/logs",
        "log_iterations": 100,
        "max_epochs": 100,
        "augment": 1,
        "backbone": "resnet34",
        "dropout": 0.5
    },
    "tune": {
        "optimizer": ["sgd", "adam"],
        "lr": [1e-4, 1e-3, 1e-2],
        "weight_decay": [5e-5, 5e-4, 5e-3]
    }
}
"""
import argparse
import json
import itertools

from pathlib import Path


def _product(dict_list):
    keys = dict_list.keys()
    vals = dict_list.values()

    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def _load(path):
    with open(path, 'r') as f:
        template = json.load(f)

    assert 'fixed' in template
    assert 'tune' in template
    assert 'log_dir' in template['fixed']

    return template


def main(params):
    template = _load(params.json)

    setup = template.get('setup', '')
    fixed = template['fixed']
    tune = template['tune']

    log_dir = Path(fixed.pop('log_dir'))

    runs = list()

    for config in _product(tune):
        config['log_dir'] = str(log_dir / '_'.join('%s%s' % (k, v) for k, v in config.items()))
        config.update(fixed)

        line = [template['command']]

        for key, val in config.items():
            line.append('--%s %s' % (key, val))

        run = ' \\\n    '.join(line)
        runs.append(run)

    output_dir = Path(params.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(params.splits):
        text = ['\n'.join(setup)]

        for idx in range(i * len(runs) // params.splits, (i+1) * len(runs) // params.splits):
            text.append(runs[idx])

        output_file = output_dir / ('%s.sh' % i)
        output_file.write_text('\n\n'.join(text))

        print('%s has %d runs' % (output_file, len(text) - 1))

    print('Generated %d total runs.' % len(runs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--splits', type=int, default=1)

    main(parser.parse_args())
