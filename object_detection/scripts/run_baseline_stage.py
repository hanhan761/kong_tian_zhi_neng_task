"""Train the unchanged baseline architecture and save the exact launch record."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='01_baseline_s640_closemosaic_80e')
    parser.add_argument('--weights', default='runs/00_历史普通模型_90.26/weights/best.pt')
    parser.add_argument('--img', default=640, type=int)
    parser.add_argument('--batch', default=8, type=int)
    parser.add_argument('--epochs', default=80, type=int)
    parser.add_argument('--patience', default=30, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--hyp', default='data/hyps/hyp.baseline-closemosaic.yaml')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    command = [sys.executable, '-u', 'train.py', '--weights', args.weights, '--cfg-from-weights',
               '--data', 'yolov5_project.yaml', '--hyp', args.hyp, '--img', str(args.img),
               '--batch-size', str(args.batch), '--epochs', str(args.epochs),
               '--patience', str(args.patience), '--seed', str(args.seed), '--device', '0',
               '--workers', '0', '--cache', 'disk', '--optimizer', 'SGD',
               '--selection-metric', 'map50', '--save-period', '10',
               '--project', 'runs/01_普通模型达标训练', '--name', args.name]
    print(subprocess.list2cmdline(command), flush=True)
    if args.dry_run:
        return
    output = ROOT / 'reports/01_普通模型达标训练' / args.name
    output.mkdir(parents=True, exist_ok=False)
    if (ROOT / 'runs/01_普通模型达标训练' / args.name).exists():
        raise FileExistsError('Run already exists; choose a new --name.')
    env = os.environ.copy()
    env.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8', GIT_PYTHON_REFRESH='quiet',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', WANDB_DISABLED='true')
    record = dict(started=datetime.now().astimezone().isoformat(), command=command, cwd=str(ROOT),
                  start_weights_sha256=hashlib.sha256((ROOT/args.weights).read_bytes()).hexdigest(),
                  intended_change='Training recipe only; unchanged ordinary YOLOv5s architecture',
                  status='launching')
    path = output / 'launch.json'
    def save():
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    save()
    with (output / 'console.log').open('w', encoding='utf-8') as stream:
        proc = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=stream, stderr=subprocess.STDOUT)
        record.update(pid=proc.pid, status='running')
        save()
        print('Training PID:', proc.pid, 'Log:', str(output / 'console.log'), flush=True)
        code = proc.wait()
    record.update(returncode=code, status='finished' if code == 0 else 'failed',
                  ended=datetime.now().astimezone().isoformat())
    save()
    print(json.dumps(record, ensure_ascii=True), flush=True)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
