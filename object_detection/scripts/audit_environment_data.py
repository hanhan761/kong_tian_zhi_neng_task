"""Check the installed environment and audit the course data without changing it."""
from collections import Counter
from datetime import datetime
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT/'reports/00_环境与数据核查'/datetime.now().strftime('%Y%m%d_%H%M%S')
    output.mkdir(parents=True, exist_ok=False)
    modules = {}
    for name in ('torch','torchvision','PIL','cv2','numpy','yaml','matplotlib','pandas','seaborn','scipy','psutil','tqdm'):
        try:
            module = importlib.import_module(name)
            modules[name] = dict(importable=True, version=getattr(module,'__version__','unknown'))
        except Exception as exc:
            modules[name] = dict(importable=False,error=str(exc))
    import torch
    from PIL import Image
    checks = dict(python38=sys.version_info[:2]==(3,8), torch24=torch.__version__.startswith('2.4.'),
                  torch_cuda118=torch.version.cuda=='11.8', pillow911=modules['PIL'].get('version')=='9.1.1',
                  cuda_available=torch.cuda.is_available(), imports_ok=all(x['importable'] for x in modules.values()))
    environment = dict(timestamp=datetime.now().astimezone().isoformat(), python=sys.executable,
                       python_version=sys.version, platform=platform.platform(), modules=modules,
                       cuda_runtime=torch.version.cuda,
                       gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                       checks=checks, all_required_checks_pass=all(checks.values()))
    (output/'environment.json').write_text(json.dumps(environment,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'requirements-installed.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),encoding='utf-8')
    records=[]
    summaries={}
    examples={}
    for split in ('train','val'):
        counts=Counter()
        backups=0
        split_records=[]
        for path in sorted((ROOT/'mydata/images'/split).iterdir()):
            if path.suffix.lower() not in {'.jpg','.jpeg','.png','.bmp'}:
                continue
            label=ROOT/'mydata/labels'/split/(path.stem+'.txt')
            if '(1)' in path.stem:
                plain=path.with_name(path.name.replace('(1)',''))
                plain_label=label.with_name(label.name.replace('(1)',''))
                if (plain.exists() and plain_label.exists() and plain.read_bytes()==path.read_bytes()
                        and plain_label.read_bytes()==label.read_bytes()):
                    backups+=1
                    continue
            lines=[line.split() for line in label.read_text().splitlines() if line.strip()]
            for line in lines:
                if len(line)!=5:
                    raise ValueError(str(label))
                class_id=int(line[0])
                box=list(map(float,line[1:]))
                if class_id not in range(5) or not all(0<=x<=1 for x in box) or min(box[2:])<=0:
                    raise ValueError(str(label))
                counts[class_id]+=1
            with Image.open(path) as image:
                size=image.size
            record=dict(split=split,file=str(path.relative_to(ROOT)),width=size[0],height=size[1],
                        image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        labels_sha256=hashlib.sha256(label.read_bytes()).hexdigest(),
                        boxes=len(lines))
            split_records.append(record)
            for line in lines:
                c=int(line[0])
                examples.setdefault(str(c),dict(image=record['file'],label=str(label.relative_to(ROOT)),
                                               image_size=size,raw_line=' '.join(line)))
        summaries[split]=dict(images=len(split_records),boxes=sum(counts.values()),
                              per_class=dict(counts),identical_backups_skipped=backups)
        records.extend(split_records)
    a={x['image_sha256'] for x in records if x['split']=='train'}
    b={x['image_sha256'] for x in records if x['split']=='val'}
    audit=dict(splits=summaries,exact_image_hash_overlap=len(a&b),
               class_names=['four rotor','six rotor','fixed wing','birds','balloon'],
               examples=examples, policy='Identical backup image AND label pairs are omitted; all other items kept.')
    (output/'course_dataset.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'course_manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(output),environment=environment,audit=audit),ensure_ascii=True,indent=2))


if __name__=='__main__':
    main()
