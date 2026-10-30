"""Create a fresh precision-search experiment; train calibration, validation only."""
import argparse
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).absolute().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--max-full', type=int, default=7)
    parser.add_argument('--screen-per-class', type=int, default=32)
    a = parser.parse_args()
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    out = (a.out or root / 'runs' / ('precision_search_' + stamp)).absolute()
    if out.exists():
        parser.error('Output exists; use a new directory to preserve prior experiments')
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES='-1', TEMP=str(root / '.tmp'), TMP=str(root / '.tmp'), TF_CPP_MIN_LOG_LEVEL='3')
    steps = [
        ['prepare_precision_search.py', '--out', str(out), '--screen-per-class', str(a.screen_per_class)],
        ['search_kwt_precision.py', '--run', str(out), '--max-full', str(a.max_full)],
        ['search_precision_per_channel.py', '--parent', str(out)],
        ['export_precision_common_payload.py', '--run', str(out)],
        ['summarize_precision_search.py', '--run', str(out)],
        ['verify_precision_search.py', '--run', str(out)],
    ]
    for step in steps:
        command = [sys.executable, '-B', '-u'] + step
        if step[0] == 'verify_precision_search.py':
            verified = subprocess.run(command, cwd=str(root), env=env, check=True,
                                      stdout=subprocess.PIPE, universal_newlines=True,
                                      encoding='utf-8')
            (out / 'independent_verification.json').write_text(verified.stdout, encoding='utf-8')
            print(verified.stdout, end='')
        else:
            subprocess.run(command, cwd=str(root), env=env, check=True)
    print('COMPLETED_EXPERIMENT=' + str(out))


if __name__ == '__main__':
    main()
