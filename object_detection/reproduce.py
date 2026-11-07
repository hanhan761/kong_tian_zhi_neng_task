"""Reproduce the saved course experiments from an independently copied project."""
import argparse
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
STAGES = {
    'baseline': '01_普通模型达标训练/01_baseline_s640_closemosaic_80e',
    'separation': '03_专业数据训练/13_separation_v2_coco_s640_60e',
    'final': '02_模型优化/22_yolov5l_coco_s640_120e',
}

def relocate(value):
    text = str(value).replace(chr(92), '/')
    for old in ['E:/yolo目标检测2/01_实验工程/', 'E:/yolo目标检测/01_实验项目/']:
        if text.lower().startswith(old.lower()):
            text = text[len(old):]
            break
    path = Path(text)
    candidate = path if path.is_absolute() else ROOT / path
    if candidate.is_file():
        return str(candidate)
    try:
        relative = candidate.relative_to(ROOT / 'runs')
    except ValueError:
        return str(candidate)
    groups = {
        '01_普通模型达标训练': '01_基础复现',
        '03_专业数据训练': '02_自制数据集',
        '02_模型优化': '03_模型优化',
    }
    if relative.parts and relative.parts[0] in groups:
        fallback = ROOT.parent / groups[relative.parts[0]] / 'runs' / 'train' / Path(*relative.parts[1:])
        if fallback.is_file():
            return str(fallback)
    return str(candidate)

def stage_result_path(stage):
    folder = {'baseline': '01_基础复现', 'separation': '02_自制数据集', 'final': '03_模型优化'}[stage]
    run_name = Path(STAGES[stage]).name
    return ROOT.parent / folder / 'runs' / 'train' / run_name

def prepare():
    import yaml
    folder = ROOT / 'separation_dataset_ready1000'
    folder.mkdir(parents=True, exist_ok=True)
    base = dict(path=folder.as_posix(), train='images/train', val='images/val',
                test='images/test', nc=2, names=['separation_component', 'flash_smoke'])
    for name, split in [('runtime.yaml', 'images/val'), ('runtime_test.yaml', 'images/test')]:
        config = dict(base, val=split)
        (folder / name).write_text(yaml.safe_dump(config, allow_unicode=False, sort_keys=False), encoding='ascii')
    print('Configured dataset:', folder, flush=True)

def selected(args):
    return args.run or STAGES[args.stage]

def config_for(args):
    import yaml
    run = (ROOT / 'runs' / selected(args)).resolve()
    try:
        run.relative_to((ROOT / 'runs').resolve())
    except ValueError:
        raise ValueError('The selected run must be inside this project runs directory.')
    opt = yaml.safe_load((run / 'opt.yaml').read_text(encoding='utf-8'))
    return run, opt

def training_command(args):
    import yaml
    run, opt = config_for(args)
    launch_path = ROOT / 'reports' / selected(args) / 'launch.json'
    if launch_path.is_file():
        original = json.loads(launch_path.read_text(encoding='utf-8'))['command']
        train_index = next(i for i,x in enumerate(original) if Path(x).name == 'train.py')
        cmd = [sys.executable, '-u', 'train.py'] + original[train_index + 1:]
        for flag in ['--weights', '--hyp', '--cfg', '--data']:
            if flag not in cmd:
                continue
            index = cmd.index(flag) + 1
            value = cmd[index]
            if flag == '--data' and 'separation_dataset' in value:
                value = 'separation_dataset_ready1000/runtime.yaml'
            cmd[index] = relocate(value)
            if not Path(cmd[index]).is_file():
                raise FileNotFoundError('Missing original launch input: ' + cmd[index])
        for flag, value in [('--project', str(ROOT / 'runs/reproduced')), ('--name', args.name),
                            ('--device', args.device), ('--epochs', args.epochs), ('--batch-size', args.batch)]:
            if value is None:
                continue
            if flag in cmd:
                cmd[cmd.index(flag)+1] = str(value)
            else:
                cmd.extend([flag, str(value)])
        print('Recipe source:', launch_path, flush=True)
        return cmd
    weights = relocate(opt['weights'])
    if not Path(weights).is_file():
        raise FileNotFoundError('Required starting checkpoint is missing: ' + weights)
    dataset = opt['data']
    if 'separation_dataset' in str(dataset):
        dataset = 'separation_dataset_ready1000/runtime.yaml'
    cfg = opt.get('cfg', '')
    if isinstance(cfg, dict):
        config_dir = ROOT / 'reproduction_configs'
        config_dir.mkdir(exist_ok=True)
        cfg_path = config_dir / (args.name + '_model.yaml')
        cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=False, sort_keys=False), encoding='ascii')
        cfg = str(cfg_path)
    cmd = [sys.executable, '-u', 'train.py', '--weights', weights,
           '--data', relocate(dataset), '--hyp', str(run / 'hyp.yaml'),
           '--img', str(opt['imgsz']), '--batch-size', str(args.batch or opt['batch_size']),
           '--epochs', str(args.epochs or opt['epochs']), '--patience', str(opt['patience']),
           '--seed', str(opt['seed']), '--device', args.device, '--workers', '0',
           '--optimizer', opt['optimizer'], '--selection-metric', opt.get('selection_metric', 'map50'),
           '--project', str(ROOT / 'runs/reproduced'), '--name', args.name]
    if cfg:
        cmd.extend(['--cfg', relocate(cfg)])
    if opt.get('cfg_from_weights'):
        cmd.append('--cfg-from-weights')
    for flag in ['cos_lr', 'image_weights', 'rect', 'multi_scale', 'single_cls', 'noautoanchor']:
        if opt.get(flag):
            cmd.append('--' + flag.replace('_', '-'))
    if opt.get('close_mosaic', 0):
        cmd.extend(['--close-mosaic', str(opt['close_mosaic'])])
    freeze = opt.get('freeze')
    if freeze and freeze != [0]:
        freeze = freeze if isinstance(freeze, list) else [freeze]
        cmd.extend(['--freeze'] + [str(x) for x in freeze])
    if opt.get('cache'):
        cmd.extend(['--cache', str(opt['cache'])])
    if opt.get('save_period'):
        cmd.extend(['--save-period', str(opt['save_period'])])
    return cmd

def execute(cmd, args):
    print(subprocess.list2cmdline(cmd), flush=True)
    if args.dry_run:
        return
    report = ROOT / 'reproduction_logs' / args.name
    if report.exists():
        raise FileExistsError('Choose a new --name; the log already exists: ' + str(report))
    if args.action == 'train' and (ROOT / 'runs/reproduced' / args.name).exists():
        raise FileExistsError('Choose a new --name; the run already exists.')
    report.mkdir(parents=True)
    env = os.environ.copy()
    env.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8', GIT_PYTHON_REFRESH='quiet',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', WANDB_DISABLED='true')
    record = dict(command=cmd, cwd=str(ROOT), started=datetime.now().astimezone().isoformat())
    path = report / 'command.json'
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    with (report / 'console.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, encoding='utf-8', errors='replace')
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    record.update(returncode=code, ended=datetime.now().astimezone().isoformat())
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    raise SystemExit(code)

def check():
    import yaml
    import torch
    import torchvision
    import cv2
    data = yaml.safe_load((ROOT / 'yolov5_project.yaml').read_text(encoding='utf-8'))
    for split in ['train', 'val']:
        files = [p for p in (ROOT / data[split]).iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        kept, duplicates = [], 0
        for image in files:
            original = image.with_name(image.stem[:-3] + image.suffix) if image.stem.endswith('(1)') else None
            label = ROOT / 'mydata/labels' / split / (image.stem + '.txt')
            original_label = ROOT / 'mydata/labels' / split / (original.stem + '.txt') if original else None
            if (original and original.is_file() and label.is_file() and original_label.is_file()
                    and image.read_bytes() == original.read_bytes() and label.read_bytes() == original_label.read_bytes()):
                duplicates += 1
            else:
                kept.append(image)
        print('Course', split, 'raw:', len(files), 'identical backups:', duplicates, 'used:', len(kept))
        assert len(kept) == {'train': 904, 'val': 903}[split]
    for stage, rel in STAGES.items():
        recipe = ROOT / 'runs' / rel
        opt = yaml.safe_load((recipe / 'opt.yaml').read_text(encoding='utf-8'))
        result = stage_result_path(stage)
        assert (result / 'weights/best.pt').is_file(), str(result / 'weights/best.pt')
        assert Path(relocate(opt['weights'])).is_file(), opt['weights']
        assert (recipe / 'hyp.yaml').is_file(), stage
    folder = ROOT / 'separation_dataset_ready1000'
    missing = [str(folder / 'images' / split) for split in ['train', 'val', 'test']
               if not (folder / 'images' / split).is_dir()]
    missing += [str(folder / 'labels' / split) for split in ['train', 'val', 'test']
                if not (folder / 'labels' / split).is_dir()]
    if missing:
        raise FileNotFoundError(
            'The Kaggle dataset is not installed. Download spacecraft-separation-detection-1000, '
            'extract its images and labels folders into ' + str(folder) +
            ', then run "python reproduce.py prepare" and "python reproduce.py check". '
            'Public Kaggle URL: https://www.kaggle.com/datasets/martinqu/spacecraft-separation-detection-1000. Missing: ' + ', '.join(missing))
    for split, wanted in [('train', 700), ('val', 150), ('test', 150)]:
        images = list((folder / 'images' / split).glob('*.jpg'))
        labels = list((folder / 'labels' / split).glob('*.txt'))
        assert len(images) == len(labels) == wanted, split
        assert {p.stem for p in images} == {p.stem for p in labels}, split
    print(json.dumps(dict(python=sys.version, torch=torch.__version__, torchvision=torchvision.__version__,
        opencv=cv2.__version__, cuda_available=torch.cuda.is_available(),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        required_inputs_present=True), ensure_ascii=False, indent=2))

def build_data(args):
    spec = importlib.util.spec_from_file_location('rebuild_separation', ROOT / 'scripts/build_separation_v2.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SOURCE = ROOT / 'source_media/NASA分离与火工公开影像'
    module.DATA = ROOT / 'reconstructed_data' / args.name
    module.REPORT = ROOT / 'reconstructed_data' / (args.name + '_report')
    if module.DATA.exists() or module.REPORT.exists():
        raise FileExistsError('Use a new dataset name; supplied data will not be overwritten.')
    print('Input videos:', module.SOURCE)
    print('Output images/labels:', module.DATA)
    if args.dry_run:
        return
    module.DATA.parent.mkdir(parents=True, exist_ok=True)
    module.REPORT.mkdir(parents=True, exist_ok=True)
    sys.argv = [str(ROOT / 'scripts/build_separation_v2.py')]
    module.main()
    import yaml
    config=dict(path=module.DATA.as_posix(),train='images/train',val='images/val',test='images/test',
                nc=2,names=['separation_component','flash_smoke'])
    for filename,split in [('runtime.yaml','images/val'),('runtime_test.yaml','images/test')]:
        (module.DATA/filename).write_text(yaml.safe_dump(dict(config,val=split),allow_unicode=False,sort_keys=False),encoding='ascii')
    print('Rebuilt data configuration:',module.DATA/'runtime.yaml')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'check', 'train', 'validate', 'rebuild-data'])
    parser.add_argument('--stage', choices=list(STAGES), default='baseline')
    parser.add_argument('--run', help='Original run directory relative to runs/, for exploration recipes')
    parser.add_argument('--name', default=datetime.now().strftime('replay_%Y%m%d_%H%M%S'))
    parser.add_argument('--device', default='0')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if any(x in args.name for x in ['/', chr(92), '..', ':']):
        parser.error('Use a simple new run name without path separators.')
    os.chdir(str(ROOT))
    if args.action == 'prepare':
        prepare()
    elif args.action == 'check':
        prepare()
        check()
    elif args.action == 'rebuild-data':
        build_data(args)
    else:
        prepare()
        if args.action == 'train':
            execute(training_command(args), args)
        else:
            if args.split == 'test' and args.stage != 'separation':
                parser.error('The course dataset has no independent test split; select val.')
            run, opt = config_for(args)
            dataset = 'yolov5_project.yaml'
            if args.stage == 'separation':
                dataset = 'separation_dataset_ready1000/' + ('runtime_test.yaml' if args.split == 'test' else 'runtime.yaml')
            weights = stage_result_path(args.stage) if not args.run else Path(relocate(run / 'weights/best.pt'))
            weight_file = weights / 'weights' / 'best.pt' if weights.is_dir() else weights
            if not weight_file.is_file():
                raise FileNotFoundError('Saved best checkpoint is not present in the submission package: ' + str(weight_file))
            execute([sys.executable, '-u', 'scripts/validate_checkpoint.py', '--weights', str(weight_file),
                '--data', str(ROOT / dataset), '--img', '640', '--batch', '8', '--device', args.device,
                '--output', str(ROOT / 'validation_reproduced' / args.name)], args)

if __name__ == '__main__':
    main()
