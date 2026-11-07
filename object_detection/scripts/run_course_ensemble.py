"""Combine course detectors by concatenating predictions before standard NMS.

The resulting .pt contains all component models, so inference costs more. This
is prediction ensembling, not a newly trained single detector or weight average.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_weight_average import BASE, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', nargs='+', required=True)
    parser.add_argument('--name', default='08_ensemble_baseline_ca')
    args = parser.parse_args()
    if len(args.weights) < 2:
        parser.error('Use at least two models')
    os.chdir(str(ROOT))
    os.environ.setdefault('GIT_PYTHON_REFRESH', 'quiet')
    import torch
    from models.experimental import attempt_load
    torch.set_num_threads(4)
    sources = [(ROOT / p).resolve() for p in args.weights]
    report = ROOT / 'reports/02_模型优化' / args.name
    run = ROOT / 'runs/02_模型优化' / args.name
    if report.exists() or run.exists():
        raise FileExistsError('Choose a new --name; existing evidence is preserved.')
    model = attempt_load(sources, device=torch.device('cpu'), fuse=False).eval()
    if any(component.names != model.names for component in model):
        raise ValueError('Class names and order must match')
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    report.mkdir(parents=True)
    (run / 'weights').mkdir(parents=True)
    destination = run / 'weights/ensemble.pt'
    torch.save(dict(model=model.half(), ema=None, optimizer=None, epoch=-1,
                    best_fitness=None, updates=None, date=datetime.now().isoformat()), destination)
    record = dict(started=datetime.now().astimezone().isoformat(), method='prediction_ensemble',
                  command=sys.argv, sources=[dict(path=str(p), sha256=sha(p)) for p in sources],
                  output=str(destination), output_sha256=sha(destination),
                  component_count=len(sources), parameters=sum(p.numel() for p in model.parameters()),
                  note='Concatenate all model predictions, then use the unchanged class-aware NMS. No extra training.',
                  status='validating')
    write(report / 'launch.json', record)
    command = [sys.executable, '-u', 'scripts/validate_checkpoint.py', '--weights', str(destination),
               '--data', 'yolov5_project.yaml', '--img', '640', '--batch', '8', '--device', '0',
               '--output', str(report / 'independent_validation_final')]
    env = os.environ.copy()
    env.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8', GIT_PYTHON_REFRESH='quiet')
    with (report / 'independent_validation_console.log').open('w', encoding='utf-8') as stream:
        code = subprocess.run(command, cwd=str(ROOT), env=env, stdout=stream,
                              stderr=subprocess.STDOUT).returncode
    record.update(status='finished' if code == 0 else 'failed', returncode=code,
                  ended=datetime.now().astimezone().isoformat())
    write(report / 'launch.json', record)
    if code:
        raise SystemExit(code)
    baseline = json.loads(BASE.read_text(encoding='utf-8'))
    candidate = json.loads((report / 'independent_validation_final/metrics.json').read_text(encoding='utf-8'))
    keys = ['images', 'instances', 'image_size', 'batch_size', 'half', 'validated_paths_sha256',
            'data_yaml_sha256', 'confidence_threshold', 'nms_iou_threshold', 'max_detections',
            'augmented_inference']
    comparison = dict(baseline_map50=baseline['map50'], candidate_map50=candidate['map50'],
                      same_validation_protocol=all(baseline[k] == candidate[k] for k in keys),
                      improvement_percentage_points=100 * (candidate['map50'] - baseline['map50']),
                      improved=candidate['map50'] > baseline['map50'],
                      deployment_note='Contains multiple detectors; compare parameter count and inference speed as well as AP.')
    write(report / 'comparison.json', comparison)
    print(json.dumps(comparison, ensure_ascii=True, indent=2), flush=True)


if __name__ == '__main__':
    main()
