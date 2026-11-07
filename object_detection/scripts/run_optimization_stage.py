"""Run a separately recorded P2 + Coordinate Attention optimization experiment."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='02_p2_coordatt_s640_120e')
    parser.add_argument('--weights', default='runs/01_普通模型达标训练/01_baseline_s640_closemosaic_80e/weights/best.pt')
    parser.add_argument('--cfg', default='models/yolov5s-p2-course-coordatt.yaml')
    parser.add_argument('--hyp', default='data/hyps/hyp.course-p2-640-refine.yaml')
    parser.add_argument('--img', default=640, type=int)
    parser.add_argument('--batch', default=8, type=int)
    parser.add_argument('--epochs', default=120, type=int)
    parser.add_argument('--patience', default=40, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    command = [sys.executable, '-u', 'train.py', '--weights', args.weights,
               '--cfg', args.cfg, '--data', 'yolov5_project.yaml', '--hyp', args.hyp,
               '--img', str(args.img), '--batch-size', str(args.batch),
               '--epochs', str(args.epochs), '--patience', str(args.patience),
               '--seed', str(args.seed), '--device', '0', '--workers', '0',
               '--cache', 'disk', '--optimizer', 'SGD', '--selection-metric', 'map50',
               '--save-period', '10', '--project', 'runs/02_模型优化', '--name', args.name]
    print(subprocess.list2cmdline(command), flush=True)
    if args.dry_run:
        return
    output = ROOT / 'reports/02_模型优化' / args.name
    output.mkdir(parents=True, exist_ok=False)
    run_dir = ROOT / 'runs/02_模型优化' / args.name
    if run_dir.exists():
        raise FileExistsError('Run already exists; choose a new --name.')
    env = os.environ.copy()
    env.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8', GIT_PYTHON_REFRESH='quiet',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', WANDB_DISABLED='true')
    record = dict(started=datetime.now().astimezone().isoformat(), command=command,
                  cwd=str(ROOT), start_weights=str((ROOT / args.weights).resolve()),
                  start_weights_sha256=sha256(ROOT / args.weights), cfg=args.cfg, hyp=args.hyp,
                  intended_change='Four-scale P2 output plus identity-initialized Coordinate Attention',
                  status='launching')
    path = output / 'launch.json'
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    with (output / 'console.log').open('w', encoding='utf-8') as stream:
        proc = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=stream, stderr=subprocess.STDOUT)
        record.update(pid=proc.pid, status='running')
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        print('Training PID:', proc.pid, 'Log:', str(output / 'console.log'), flush=True)
        code = proc.wait()
    record.update(returncode=code, status='finished' if code == 0 else 'failed',
                  ended=datetime.now().astimezone().isoformat())
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(record, ensure_ascii=True), flush=True)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
