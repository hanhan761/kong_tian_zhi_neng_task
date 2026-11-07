"""Average two compatible course checkpoints, then evaluate at the fixed protocol.

This creates one model, not an ensemble. The 50:50 ratio is specified before
validation. Floating buffers are averaged too; integer counters use the maximum.
"""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = ROOT / 'reports/01_普通模型达标训练/01_baseline_s640_closemosaic_80e/independent_validation_20261022/metrics.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights-a', required=True)
    parser.add_argument('--weights-b', required=True)
    parser.add_argument('--alpha', type=float, default=0.5, help='Fraction of weights B')
    parser.add_argument('--name', default='07_average_ca_nwd_50_50')
    args = parser.parse_args()
    if not 0 < args.alpha < 1:
        parser.error('--alpha must be between zero and one')
    os.chdir(str(ROOT))
    os.environ.setdefault('GIT_PYTHON_REFRESH', 'quiet')
    import torch
    torch.set_num_threads(4)
    sources = [(ROOT / p).resolve() for p in [args.weights_a, args.weights_b]]
    report = ROOT / 'reports/02_模型优化' / args.name
    run = ROOT / 'runs/02_模型优化' / args.name
    if report.exists() or run.exists():
        raise FileExistsError('Choose a new --name; existing evidence is preserved.')
    models = []
    for path in sources:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        models.append((checkpoint.get('ema') or checkpoint['model']).float().eval())
    a, b = models
    if a.names != b.names or a.yaml != b.yaml:
        raise ValueError('Class order and model configuration must be identical.')
    sa, sb = a.state_dict(), b.state_dict()
    if sa.keys() != sb.keys() or any(sa[k].shape != sb[k].shape for k in sa):
        raise ValueError('State dictionaries must have identical names and shapes.')
    for key in sa:
        if 'anchors' in key and not torch.equal(sa[key], sb[key]):
            raise ValueError('Anchor geometry must match: ' + key)
    averaged = deepcopy(a)
    # The untrained identity-CA bridge has yaml/names but training has not yet
    # attached the convenience nc attribute used by val.py's class-count check.
    averaged.nc = len(a.names)
    state = {}
    for key in sa:
        if torch.is_floating_point(sa[key]):
            state[key] = (1 - args.alpha) * sa[key] + args.alpha * sb[key]
        else:
            state[key] = torch.maximum(sa[key], sb[key])
    averaged.load_state_dict(state, strict=True)
    for parameter in averaged.parameters():
        parameter.requires_grad_(False)
    report.mkdir(parents=True)
    (run / 'weights').mkdir(parents=True)
    destination = run / 'weights/averaged.pt'
    torch.save(dict(model=averaged.half(), ema=None, optimizer=None, epoch=-1,
                    best_fitness=None, updates=None, date=datetime.now().isoformat()), destination)
    record = dict(started=datetime.now().astimezone().isoformat(), method='weight_average',
                  command=sys.argv, sources=[dict(path=str(p), sha256=sha(p)) for p in sources],
                  fraction_b=args.alpha, output=str(destination), output_sha256=sha(destination),
                  tensors=len(state), parameters=sum(p.numel() for p in averaged.parameters()),
                  integer_buffer_policy='elementwise maximum',
                  floating_buffer_policy='same convex average as parameters; no BN recalibration',
                  note='Ratio specified explicitly before validation; no automatic ratio grid search; no additional training.',
                  status='validating')
    write(report / 'launch.json', record)
    env = os.environ.copy()
    env.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8', GIT_PYTHON_REFRESH='quiet')
    command = [sys.executable, '-u', 'scripts/validate_checkpoint.py', '--weights', str(destination),
               '--data', 'yolov5_project.yaml', '--img', '640', '--batch', '8', '--device', '0',
               '--output', str(report / 'independent_validation_final')]
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
                      improved=candidate['map50'] > baseline['map50'])
    write(report / 'comparison.json', comparison)
    print(json.dumps(comparison, ensure_ascii=True, indent=2), flush=True)


if __name__ == '__main__':
    main()
