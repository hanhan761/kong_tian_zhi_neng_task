"""Train the two-class public separation-event dataset with a separate record."""
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
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='03_separation_yolov5s_s640_60e')
    parser.add_argument('--weights', default='runs/01_普通模型达标训练/01_baseline_s640_closemosaic_80e/weights/best.pt')
    parser.add_argument('--cfg', default='models/yolov5s.yaml')
    parser.add_argument('--data', default='separation_dataset_1000/separation_dataset.yaml')
    parser.add_argument('--hyp', default='data/hyps/hyp.baseline-closemosaic.yaml')
    parser.add_argument('--img', default=640, type=int)
    parser.add_argument('--batch', default=8, type=int)
    parser.add_argument('--epochs', default=60, type=int)
    parser.add_argument('--patience', default=20, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--freeze', default=0, type=int, help='Freeze the first N model layers; 0 trains all layers')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--cos-lr', action='store_true')
    parser.add_argument('--validate-after', action='store_true', help='Validate best on val and test after training; never select weights using test')
    args = parser.parse_args()
    command = [sys.executable, '-u', 'train.py', '--weights', args.weights, '--cfg', args.cfg,
               '--data', args.data, '--hyp', args.hyp, '--img', str(args.img),
               '--batch-size', str(args.batch), '--epochs', str(args.epochs),
               '--patience', str(args.patience), '--seed', str(args.seed), '--device', '0',
               '--workers', '0', '--cache', 'disk', '--optimizer', 'SGD', '--selection-metric', 'map50',
               '--save-period', '10', '--project', 'runs/03_专业数据训练', '--name', args.name]
    if args.cos_lr:
        command.append('--cos-lr')
    if args.freeze:
        if args.freeze < 0:
            parser.error('--freeze must be nonnegative')
        command.extend(['--freeze', str(args.freeze)])
    print(subprocess.list2cmdline(command), flush=True)
    if args.dry_run:
        return
    output = ROOT / 'reports/03_专业数据训练' / args.name
    output.mkdir(parents=True, exist_ok=False)
    run_dir = ROOT / 'runs/03_专业数据训练' / args.name
    if run_dir.exists():
        raise FileExistsError('Run already exists; choose a new --name.')
    from freeze_dataset_labels import freeze
    snapshot = freeze((ROOT / args.data).parent, output / 'training_labels_snapshot.zip')
    env = os.environ.copy()
    env.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8', GIT_PYTHON_REFRESH='quiet',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', WANDB_DISABLED='true')
    record = dict(started=datetime.now().astimezone().isoformat(), command=command, cwd=str(ROOT),
                  start_weights=str((ROOT / args.weights).resolve()), start_weights_sha256=sha256(ROOT / args.weights),
                  data=args.data, cfg=args.cfg, hyp=args.hyp,
                  data_yaml_sha256=sha256(ROOT / args.data), hyp_sha256=sha256(ROOT / args.hyp),
                  dataset_snapshot=snapshot,
                  intended_change='Two-class separation_component and flash_smoke transfer learning', status='launching')
    path = output / 'launch.json'
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    with (output / 'console.log').open('w', encoding='utf-8') as stream:
        proc = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=stream, stderr=subprocess.STDOUT)
        record.update(pid=proc.pid, status='running')
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        print('Training PID:', proc.pid, 'Log:', str(output / 'console.log'), flush=True)
        code = proc.wait()
    record.update(returncode=code, status='finished' if code == 0 else 'failed', ended=datetime.now().astimezone().isoformat())
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(record, ensure_ascii=True), flush=True)
    if code == 0 and args.validate_after:
        for split in ['val', 'test']:
            data = ROOT / args.data
            if split == 'test':
                data = data.with_name(data.stem + '_test.yaml')
            if not data.is_file():
                raise FileNotFoundError(data)
            command = [sys.executable, '-u', 'scripts/validate_checkpoint.py',
                       '--weights', str(run_dir / 'weights/best.pt'), '--data', str(data),
                       '--img', str(args.img), '--batch', str(args.batch), '--device', '0',
                       '--output', str(output / ('independent_validation_' + split + '_final'))]
            with (output / ('validation_' + split + '.log')).open('w', encoding='utf-8') as stream:
                subprocess.run(command, cwd=str(ROOT), env=env, stdout=stream,
                               stderr=subprocess.STDOUT, check=True)
        print('Val and test reports saved:', output, flush=True)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
