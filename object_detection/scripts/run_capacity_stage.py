"""Train a larger pretrained detector and validate at the original 640 protocol."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'reports/01_普通模型达标训练/01_baseline_s640_closemosaic_80e/independent_validation_20261022/metrics.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='17_yolov5m_coco_s640_120e')
    parser.add_argument('--weights', default='pretrained/yolov5m-v7.0.pt')
    parser.add_argument('--cfg', default='models/yolov5m.yaml')
    parser.add_argument('--hyp', default='data/hyps/hyp.course-capacity.yaml')
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--img', type=int, default=640)
    parser.add_argument('--close-mosaic', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--optimizer', choices=['SGD', 'AdamW'], default='SGD')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    os.chdir(str(ROOT))
    command = [sys.executable, '-u', 'train.py', '--weights', args.weights,
               '--cfg', args.cfg, '--data', 'yolov5_project.yaml', '--hyp', args.hyp,
               '--img', str(args.img), '--batch-size', str(args.batch),
               '--epochs', str(args.epochs), '--patience', str(args.patience),
               '--seed', str(args.seed), '--device', '0', '--workers', '0',
               '--cache', 'disk', '--optimizer', args.optimizer, '--cos-lr',
               '--close-mosaic', str(args.close_mosaic), '--selection-metric', 'map50',
               '--save-period', '20', '--project', 'runs/02_模型优化', '--name', args.name]
    print(subprocess.list2cmdline(command), flush=True)
    if args.dry_run:
        return
    report = ROOT / 'reports/02_模型优化' / args.name
    run = ROOT / 'runs/02_模型优化' / args.name
    if report.exists() or run.exists():
        raise FileExistsError('Use a new experiment name; existing artifacts are preserved.')
    report.mkdir(parents=True)
    inputs = [args.weights, args.cfg, args.hyp, 'train.py', 'yolov5_project.yaml']
    record = dict(started=datetime.now().astimezone().isoformat(), command=command,
                  cwd=str(ROOT), status='launching', method='larger_pretrained_transfer_close_mosaic',
                  input_sha256={p: digest(ROOT / p) for p in inputs},
                  intended_comparison='Fixed original validation set at 640, half, batch8, without TTA.')

    def save():
        (report / 'launch.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')

    env = os.environ.copy()
    env.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8', GIT_PYTHON_REFRESH='quiet',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', WANDB_DISABLED='true')
    save()
    with (report / 'console.log').open('w', encoding='utf-8') as stream:
        process = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=stream, stderr=subprocess.STDOUT)
        record.update(pid=process.pid, status='running')
        save()
        print('Training PID:', process.pid, flush=True)
        code = process.wait()
    record.update(returncode=code, status='trained' if code == 0 else 'failed',
                  ended=datetime.now().astimezone().isoformat())
    save()
    if code:
        raise SystemExit(code)
    output = report / 'independent_validation_final'
    validate = [sys.executable, '-u', 'scripts/validate_checkpoint.py', '--weights',
                str(run / 'weights/best.pt'), '--data', 'yolov5_project.yaml',
                '--img', '640', '--batch', '8', '--device', '0', '--output', str(output)]
    with (report / 'independent_validation_console.log').open('w', encoding='utf-8') as stream:
        result = subprocess.run(validate, cwd=str(ROOT), env=env, stdout=stream, stderr=subprocess.STDOUT)
    record.update(validation_command=validate, validation_returncode=result.returncode,
                  status='finished' if result.returncode == 0 else 'validation_failed')
    save()
    if result.returncode:
        raise SystemExit(result.returncode)
    baseline = json.loads(BASE.read_text(encoding='utf-8'))
    candidate = json.loads((output / 'metrics.json').read_text(encoding='utf-8'))
    keys = ['images', 'instances', 'image_size', 'batch_size', 'half', 'validated_paths_sha256',
            'data_yaml_sha256', 'confidence_threshold', 'nms_iou_threshold',
            'max_detections', 'augmented_inference']
    comparison = dict(baseline_metrics=str(BASE), candidate_metrics=str(output / 'metrics.json'),
                      baseline_map50=baseline['map50'], candidate_map50=candidate['map50'],
                      improvement_percentage_points=100 * (candidate['map50'] - baseline['map50']),
                      same_validation_protocol=all(baseline[k] == candidate[k] for k in keys),
                      protocol_fields=keys,
                      selection_note='Repeated validation selection; no untouched course test set.')
    (report / 'comparison.json').write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(comparison, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
