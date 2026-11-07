"""Archive the final source/config/environment record without copying datasets."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/04_交付汇总'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    archive_path = OUT / ('交付源码与环境快照_' + stamp + '.zip')
    candidates = {ROOT / name for name in ['train.py', 'val.py', 'detect.py',
                  'yolov5_project.yaml', 'requirements.txt']}
    for folder in ['models', 'utils', 'scripts', 'data/hyps']:
        for path in (ROOT / folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix in ['.py', '.yaml', '.ps1']:
                candidates.add(path)
    for folder in ['separation_dataset_1000', 'separation_dataset_v2_1000']:
        for name in ['separation_dataset.yaml', 'separation_dataset_test.yaml', 'classes.txt', 'dataset_summary.json']:
            candidates.add(ROOT / folder / name)
    candidates.add(ROOT / 'data/separation_v2_plan.json')
    for name in ['从这里开始.md', '零基础实验教程_第六版_从认识文件到VSCode复现.md',
                 '零基础实验教程_第七版_结合原图理解原理与VSCode操作.md',
                 '零基础实验教程_第八版_结合原图理解原理与VSCode操作.md',
                 '目标检测实验2.code-workspace']:
        candidates.add(ROOT.parent / name)
    candidates.update((ROOT.parent / '.vscode').glob('*.json'))
    audit = ROOT / 'reports/00_环境与数据核查/20261022_094416'
    candidates.update(audit / name for name in ['environment.json', 'requirements-installed.txt'])
    latest_audit = max((ROOT / 'reports/00_环境与数据核查').glob('*/environment.json'))
    candidates.add(latest_audit)
    candidates.add(latest_audit.with_name('requirements-installed.txt'))
    entries = []
    with zipfile.ZipFile(archive_path, 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(candidates):
            relative = path.relative_to(ROOT.parent).as_posix()
            content = path.read_bytes()
            entries.append(dict(path=relative, bytes=len(content), sha256=hashlib.sha256(content).hexdigest()))
            archive.writestr(relative, content)
        archive.writestr('source_manifest.json', json.dumps(entries, ensure_ascii=False, indent=2))
    record = dict(created=datetime.now().astimezone().isoformat(), archive=str(archive_path),
                  archive_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(), files=len(entries),
                  bytes=archive_path.stat().st_size, entries=entries,
                  note='Source/config/environment record only. Datasets, weights, videos and the Python installation remain in their documented locations.')
    (OUT / 'source_snapshot.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: record[key] for key in ['archive', 'archive_sha256', 'files', 'bytes']}, ensure_ascii=True))


if __name__ == '__main__':
    main()
