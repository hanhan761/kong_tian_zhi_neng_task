"""Copy an immutable training dataset for LabelImg edits, or refresh its index.

Refreshing hashes records current bytes; it does not certify human review.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='separation_dataset_v2_1000')
    parser.add_argument('--name', default='separation_dataset_manual_01')
    parser.add_argument('--refresh-index', action='store_true')
    args = parser.parse_args()
    if not args.name.startswith('separation_dataset_manual_') or Path(args.name).name != args.name:
        parser.error('Use a new single folder name beginning separation_dataset_manual_.')
    target = ROOT / args.name
    if not args.refresh_index:
        source = (ROOT / args.source).resolve()
        shutil.copytree(source, target, ignore=shutil.ignore_patterns('*.npy', '*.cache'))
        for name in ['separation_dataset.yaml', 'separation_dataset_test.yaml']:
            path = target / name
            data = yaml.safe_load(path.read_text(encoding='utf-8'))
            data['path'] = args.name
            path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding='utf-8')
        (target / 'manual_copy_provenance.json').write_text(json.dumps(dict(
            source=str(source), created=datetime.now().astimezone().isoformat(),
            source_manifest_sha256=sha(source / 'manifest.json'),
            strict_manual_LabelImg_annotation_complete=False,
            note='Editable copy. A copied or reindexed dataset is not proof of manual annotation.'
        ), ensure_ascii=False, indent=2), encoding='utf-8')
    elif not (target / 'manual_copy_provenance.json').is_file():
        raise ValueError('Only refresh a manual copy created by this script.')
    manifest_path = target / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if args.refresh_index:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        shutil.copy2(manifest_path, target / ('manifest_before_refresh_' + stamp + '.json'))
    counts = Counter()
    for item in manifest:
        image, label = target / item['image'], target / item['label']
        classes = []
        for line in label.read_text(encoding='utf-8').splitlines():
            if line.strip():
                classes.append(int(line.split()[0]))
        item.update(image_sha256=sha(image), label_sha256=sha(label), classes=classes,
                    annotation_method='keyframe_assisted_editable_manual_copy',
                    human_LabelImg_reviewed=False)
        counts.update(classes)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    summary_path = target / 'dataset_summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    summary.update(annotation_method='keyframe_assisted_editable_manual_copy',
                   strict_manual_LabelImg_annotation_complete=False,
                   class_boxes=dict(counts), background_images=sum(not x['classes'] for x in manifest),
                   index_refreshed=datetime.now().astimezone().isoformat(),
                   note='Hashes refreshed from current files. No image-by-image human review certification.')
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(dataset=str(target), images=len(manifest), class_boxes=dict(counts),
                          next_step='Run audit_separation_dataset.py on this dataset before training.'), ensure_ascii=True))


if __name__ == '__main__':
    main()
