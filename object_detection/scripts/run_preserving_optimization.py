"""Preserve all baseline tensors, train, and independently compare one candidate.

This script never modifies the baseline checkpoint or its validation report.
"""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASELINE = 'runs/01_普通模型达标训练/01_baseline_s640_closemosaic_80e/weights/best.pt'
BASE_METRICS = ('reports/01_普通模型达标训练/01_baseline_s640_closemosaic_80e/'
                'independent_validation_20261022/metrics.json')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def attention_checkpoint(source, output, attention_type='coordatt'):
    import torch
    import yaml
    from models.yolo import Model
    torch.set_num_threads(4)
    torch.manual_seed(42)
    ckpt = torch.load(source, map_location='cpu', weights_only=False)
    base = ckpt['model'].float().eval()
    cfg = deepcopy(base.yaml)
    # Both modules retain C3's tensor names and start as exactly identity.
    attention_class = {'coordatt': 'C3CoordAtt', 'eca': 'C3ECA'}[attention_type]
    replaced = [17, 20, 23]
    for index in replaced:
        assert cfg['head'][index - len(cfg['backbone'])][2] == 'C3'
        cfg['head'][index - len(cfg['backbone'])][2] = attention_class
    cfg['anchors'] = (base.model[-1].anchors * base.stride[:, None, None]).reshape(3, -1).tolist()
    model = Model(cfg, ch=3, nc=5).float().eval()
    original = base.state_dict()
    result = model.load_state_dict(original, strict=False)
    assert not result.unexpected_keys
    assert all('.att.' in key for key in result.missing_keys), result.missing_keys
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in original.items())
    x = torch.rand(1, 3, 128, 128)
    with torch.inference_mode():
        before = base(x)[0]
        after = model(x)[0]
    max_error = float((before - after).abs().max())
    assert torch.allclose(before, after, atol=1e-5, rtol=1e-6), max_error
    model.names = deepcopy(base.names)
    config_path = output / 'initial_model.yaml'
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    initial = output / 'initial_attention.pt'
    torch.save({'model': model.half(), 'ema': None, 'optimizer': None, 'updates': 0,
                'epoch': -1, 'best_fitness': 0.0, 'date': datetime.now().isoformat(),
                'opt': {}, 'git': {}}, initial)
    record = dict(source=str(source), source_sha256=digest(source), attention_type=attention_type,
                  checkpoint=str(initial), checkpoint_sha256=digest(initial),
                  c3_replaced_at=replaced, inherited_tensors=len(original),
                  added_tensors=len(result.missing_keys), missing_keys=result.missing_keys,
                  unexpected_keys=result.unexpected_keys,
                  every_original_tensor_identical=True, input_shape=list(x.shape),
                  initial_output_max_absolute_error=max_error,
                  initial_output_allclose=True,
                  baseline_parameters=sum(p.numel() for p in base.parameters()),
                  optimized_parameters=sum(p.numel() for p in model.parameters()))
    write_json(output / 'weight_transfer.json', record)
    return initial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='04_preserve_coordatt_s640_60e')
    parser.add_argument('--method', choices=['attention', 'refine', 'nwd'], default='attention')
    parser.add_argument('--attention-type', choices=['coordatt', 'eca'], default='coordatt',
                        help='Identity-initialized attention inserted for --method attention')
    parser.add_argument('--weights', default=BASELINE)
    parser.add_argument('--hyp', default='data/hyps/hyp.course-preserve-finetune.yaml')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--img', type=int, default=640)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--optimizer', choices=['SGD', 'Adam', 'AdamW'], default='SGD')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--nwd-ratio', type=float, default=0.1)
    parser.add_argument('--image-weights', action='store_true')
    parser.add_argument('--resume-run', action='store_true',
                        help='Resume the existing --name from last.pt, preserving its original options and logs')
    args = parser.parse_args()
    os.chdir(str(ROOT))
    output = ROOT / 'reports/02_模型优化' / args.name
    run = ROOT / 'runs/02_模型优化' / args.name
    if not args.resume_run and (output.exists() or run.exists()):
        raise FileExistsError('Use a new --name to preserve previous results, or --resume-run to continue last.pt.')
    if args.resume_run:
        import torch
        last = run / 'weights/last.pt'
        record = json.loads((output / 'launch.json').read_text(encoding='utf-8'))
        checkpoint = torch.load(last, map_location='cpu', weights_only=False)
        if checkpoint.get('optimizer') is None or checkpoint.get('epoch', -1) < 0:
            raise ValueError('last.pt has no resumable optimizer/epoch; use a new experiment name for fine-tuning.')
        snapshot = output / 'resume_snapshots' / datetime.now().strftime('%Y%m%d_%H%M%S')
        snapshot.mkdir(parents=True, exist_ok=False)
        for filename in ['last.pt', 'best.pt']:
            shutil.copy2(run / 'weights' / filename, snapshot / filename)
        resume_info = dict(resumed=datetime.now().astimezone().isoformat(),
                           checkpoint=str(last), checkpoint_sha256=digest(last),
                           last_completed_epoch_zero_based=checkpoint['epoch'],
                           snapshot_directory=str(snapshot))
        record.setdefault('resumes', []).append(resume_info)
        command = [sys.executable, '-u', 'train.py', '--resume', str(last)]
        record.update(resume_command=command, status='resuming')
        del checkpoint
    else:
        output.mkdir(parents=True)
        source = (ROOT / args.weights).resolve()
        start = attention_checkpoint(source, output, args.attention_type) if args.method == 'attention' else source
    import yaml
    if not args.resume_run:
        hyp = yaml.safe_load((ROOT / args.hyp).read_text(encoding='utf-8'))
        if args.method == 'nwd':
            hyp['nwd_ratio'] = args.nwd_ratio
        hyp_path = output / 'training_hyp.yaml'
        hyp_path.write_text(yaml.safe_dump(hyp, sort_keys=False), encoding='utf-8')
        command = [sys.executable, '-u', 'train.py', '--weights', str(start), '--cfg-from-weights',
               '--data', 'yolov5_project.yaml', '--hyp', str(hyp_path), '--img', str(args.img),
               '--batch-size', str(args.batch), '--epochs', str(args.epochs),
               '--patience', str(args.patience), '--seed', str(args.seed), '--device', '0',
               '--workers', '0', '--cache', 'disk', '--optimizer', args.optimizer, '--cos-lr',
               '--noautoanchor', '--selection-metric', 'map50', '--save-period', '10',
               '--project', 'runs/02_模型优化', '--name', args.name]
        if args.image_weights:
            command.append('--image-weights')
    env = os.environ.copy()
    env.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8', GIT_PYTHON_REFRESH='quiet',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', WANDB_DISABLED='true')
    if not args.resume_run:
        record = dict(started=datetime.now().astimezone().isoformat(), method=args.method,
                  attention_type=args.attention_type if args.method == 'attention' else None,
                  command=command, cwd=str(ROOT), start_weights=str(start),
                  start_weights_sha256=digest(start), original_baseline_sha256=digest(ROOT / BASELINE),
                  status='launching')
    launch = output / 'launch.json'
    write_json(launch, record)
    with (output / 'console.log').open('a' if args.resume_run else 'w', encoding='utf-8') as stream:
        if args.resume_run:
            stream.write('\n=== RESUMED ' + datetime.now().astimezone().isoformat() + ' ===\n')
            stream.flush()
        process = subprocess.Popen(command, cwd=str(ROOT), env=env,
                                   stdout=stream, stderr=subprocess.STDOUT)
        record.update(pid=process.pid, status='running')
        write_json(launch, record)
        print('Training PID:', process.pid, 'Log:', output / 'console.log', flush=True)
        code = process.wait()
    record.update(returncode=code, status='finished' if code == 0 else 'failed',
                  ended=datetime.now().astimezone().isoformat())
    write_json(launch, record)
    if code:
        raise SystemExit(code)
    validation = output / 'independent_validation_final'
    validation_command = [sys.executable, '-u', 'scripts/validate_checkpoint.py',
                          '--weights', str(run / 'weights/best.pt'), '--data', 'yolov5_project.yaml',
                          '--img', '640', '--batch', '8', '--device', '0', '--output', str(validation)]
    with (output / 'independent_validation_console.log').open('w', encoding='utf-8') as stream:
        subprocess.run(validation_command, cwd=str(ROOT), env=env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)
    baseline = json.loads((ROOT / BASE_METRICS).read_text(encoding='utf-8'))
    candidate = json.loads((validation / 'metrics.json').read_text(encoding='utf-8'))
    keys = ['images', 'instances', 'image_size', 'batch_size', 'half', 'validated_paths_sha256',
            'data_yaml_sha256', 'confidence_threshold', 'nms_iou_threshold',
            'max_detections', 'augmented_inference']
    same_protocol = all(baseline[key] == candidate[key] for key in keys)
    comparison = dict(baseline_metrics=str(ROOT / BASE_METRICS), candidate_metrics=str(validation / 'metrics.json'),
                      baseline_map50=baseline['map50'], candidate_map50=candidate['map50'],
                      improvement_percentage_points=(candidate['map50'] - baseline['map50']) * 100,
                      same_validation_protocol=same_protocol,
                      protocol_fields=keys, improved=candidate['map50'] > baseline['map50'],
                      selection_note='Candidates selected on the fixed course validation set; no independent course test set exists.')
    write_json(output / 'comparison.json', comparison)
    print(json.dumps(comparison, ensure_ascii=True, indent=2), flush=True)


if __name__ == '__main__':
    main()
