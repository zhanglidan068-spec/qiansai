"""Create one new validation-only progressive four-bit experiment workflow."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).absolute().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--parent', type=Path, default=ROOT / 'runs/precision_search_20261028_162338_802026')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--max-full', type=int, default=10)
    args = parser.parse_args()
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    out = (args.out or ROOT / 'runs' / ('nibble_w8a8_' + stamp)).absolute()
    if out.exists():
        parser.error('Output already exists; use a new path to preserve earlier experiments')
    (ROOT / '.tmp').mkdir(exist_ok=True)
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES='-1', TEMP=str(ROOT / '.tmp'), TMP=str(ROOT / '.tmp'),
               TF_CPP_MIN_LOG_LEVEL='3', PYTHONIOENCODING='utf-8')
    steps = [
        ('main', ['run_kwt_nibble_search.py', '--out', str(out), '--parent', str(args.parent.absolute()),
                  '--threads', str(args.threads), '--max-full', str(args.max_full)]),
        ('guarded', ['run_kwt_nibble_guarded.py', '--run', str(out), '--threads', str(args.threads)]),
        ('summary', ['summarize_nibble_experiment.py', '--run', str(out)]),
        ('audit', ['verify_nibble_experiment.py', '--run', str(out), '--output', str(out / 'independent_audit.json')]),
    ]
    for label, step in steps:
        command = [sys.executable, '-B', '-u'] + step
        log = None
        process = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, universal_newlines=True, encoding='utf-8')
        for line in process.stdout:
            print(line, end='', flush=True)
            if out.exists() and log is None:
                directory = out / 'execution_logs'
                directory.mkdir(exist_ok=True)
                log = (directory / (label + '.log')).open('w', encoding='utf-8')
            if log is not None:
                log.write(line)
                log.flush()
        return_code = process.wait()
        if log is not None:
            log.close()
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)
    support = out / 'workflow_support'
    support.mkdir(exist_ok=False)
    for name in ('run_kwt_nibble_experiments.py', 'README_四位渐进乘法运行方法.md',
                 'summarize_nibble_experiment.py', 'verify_nibble_experiment.py'):
        shutil.copyfile(ROOT / name, support / name)
    (support / 'workflow.json').write_text(json.dumps(dict(steps=steps, testing_samples_used=0),
        ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('COMPLETED_NIBBLE_EXPERIMENT=' + str(out), flush=True)


if __name__ == '__main__':
    main()
