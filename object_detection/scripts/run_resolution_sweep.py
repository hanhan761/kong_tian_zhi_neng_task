"""Evaluate fixed checkpoints with more pixels/TTA, separately from 640 inference."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--sizes', type=int, nargs='+', default=[960, 1280])
    parser.add_argument('--tta', action='store_true')
    args = parser.parse_args()
    root = ROOT / 'reports/06_新一轮优化_20261023' / args.name
    root.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8', GIT_PYTHON_REFRESH='quiet',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', WANDB_DISABLED='true')
    record = dict(started=datetime.now().astimezone().isoformat(), weights=args.weights,
                  scope='Inference computation change: same validation images/labels; batch1, larger resolution or TTA. Not a 640 single-view architecture comparison.',
                  experiments=[])
    for size in args.sizes:
        name = 'whole_s%d%s' % (size, '_tta' if args.tta else '')
        command = [sys.executable, '-u', 'scripts/validate_checkpoint.py', '--weights',
                   args.weights, '--data', 'yolov5_project.yaml', '--img', str(size),
                   '--batch', '1', '--output', str(root / name)]
        if args.tta:
            command.append('--augment')
        item = dict(name=name, command=command, status='running')
        record['experiments'].append(item)
        with (root / (name + '_console.log')).open('w', encoding='utf-8') as stream:
            process = subprocess.Popen(command, cwd=str(ROOT), env=env,
                                       stdout=stream, stderr=subprocess.STDOUT)
            item['pid'] = process.pid
            (root / 'launch.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps(dict(name=name, pid=process.pid)), flush=True)
            code = process.wait()
        item.update(returncode=code, status='finished' if code == 0 else 'failed')
        if code == 0:
            metrics = json.loads((root / name / 'metrics.json').read_text(encoding='utf-8'))
            item.update(map50=metrics['map50'], map50_95=metrics['map50_95'])
        (root / 'launch.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(item, ensure_ascii=True), flush=True)
        if code:
            raise SystemExit(code)


if __name__ == '__main__':
    main()
