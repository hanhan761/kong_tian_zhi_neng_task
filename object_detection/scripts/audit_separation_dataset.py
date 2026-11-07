"""Audit the exported separation dataset without changing any files."""
from datetime import datetime
import argparse
import hashlib
import json
import math
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'separation_dataset_1000'
REPORT_ROOT = ROOT / 'reports' / '03_专业数据集核查'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    global DATA, REPORT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='separation_dataset_1000')
    parser.add_argument('--report-name', default='')
    args = parser.parse_args()
    DATA = (ROOT / args.data).resolve()
    if args.report_name:
        REPORT_ROOT = REPORT_ROOT / args.report_name
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = REPORT_ROOT / stamp
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((DATA / 'manifest.json').read_text(encoding='utf-8'))
    by_image = {x['image'].replace('\\', '/'): x for x in manifest}
    method = json.loads((DATA / 'dataset_summary.json').read_text(encoding='utf-8'))['annotation_method']
    summary = {'manifest_entries': len(manifest), 'splits': {}, 'invalid_labels': [],
               'missing_pairs': [], 'duplicate_image_sha256': [], 'source_events': {},
               'manifest_errors': [], 'unreadable_images': [], 'events_crossing_splits': [],
               'annotation_method': method,
               'annotation_review_complete': False,
               'scope': 'File integrity and split consistency; not semantic correctness of boxes.'}
    if len(by_image) != len(manifest):
        summary['manifest_errors'].append('Duplicate image entries in manifest.')
    seen = {}
    event_splits = {}
    group_splits = {}
    observed_images = set()
    for split in ('train', 'val', 'test'):
        images = sorted((DATA / 'images' / split).glob('*.jpg'))
        # LabelImg writes a classes.txt helper; it is not an image annotation.
        labels = sorted(p for p in (DATA / 'labels' / split).glob('*.txt') if p.name != 'classes.txt')
        image_names = {p.stem for p in images}
        label_names = {p.stem for p in labels}
        for stem in sorted(image_names - label_names):
            summary['missing_pairs'].append({'split': split, 'stem': stem, 'kind': 'label'})
        for stem in sorted(label_names - image_names):
            summary['missing_pairs'].append({'split': split, 'stem': stem, 'kind': 'image'})
        class_counts = {'separation_component': 0, 'flash_smoke': 0}
        bad = []
        for label in labels:
            for line_no, line in enumerate(label.read_text(encoding='utf-8').splitlines(), 1):
                bits = line.split()
                try:
                    cls = int(bits[0]); values = [float(x) for x in bits[1:]]
                    valid = len(values) == 4 and cls in (0, 1) and all(math.isfinite(x) and 0 <= x <= 1 for x in values) and values[2] > 0 and values[3] > 0
                    if valid:
                        cx, cy, width, height = values
                        valid = (cx - width / 2 >= -1e-7 and cy - height / 2 >= -1e-7
                                 and cx + width / 2 <= 1 + 1e-7 and cy + height / 2 <= 1 + 1e-7)
                except (ValueError, IndexError):
                    valid = False; cls = -1
                if not valid:
                    bad.append({'file': str(label.relative_to(DATA)), 'line': line_no, 'text': line})
                elif cls == 0:
                    class_counts['separation_component'] += 1
                else:
                    class_counts['flash_smoke'] += 1
        for image in images:
            relative = image.relative_to(DATA).as_posix()
            observed_images.add(relative)
            digest = sha256(image)
            if digest in seen:
                summary['duplicate_image_sha256'].append({'image': str(image.relative_to(DATA)), 'same_as': seen[digest]})
            else:
                seen[digest] = str(image.relative_to(DATA))
            item = by_image.get(relative)
            if item:
                event = item['alias']; summary['source_events'].setdefault(event, {'split': split, 'images': 0, 'source_sha256': item['source_sha256']}); summary['source_events'][event]['images'] += 1
                event_splits.setdefault(event, set()).add(split)
                group_splits.setdefault(item.get('group', event), set()).add(split)
                if item.get('image_sha256') and digest != item['image_sha256']:
                    summary['manifest_errors'].append(relative + ': image hash mismatch')
                expected_label = DATA / 'labels' / split / (image.stem + '.txt')
                if item.get('label_sha256') and sha256(expected_label) != item['label_sha256']:
                    summary['manifest_errors'].append(relative + ': label hash mismatch')
                if item['split'] != split or item['label'].replace('\\', '/') != expected_label.relative_to(DATA).as_posix():
                    summary['manifest_errors'].append(relative + ': split or label path mismatch')
                try:
                    with Image.open(image) as decoded:
                        dimensions = decoded.size
                        decoded.verify()
                    if dimensions != (item['width'], item['height']):
                        summary['manifest_errors'].append(relative + ': image dimensions mismatch')
                except Exception as exc:
                    summary['unreadable_images'].append({'image': relative, 'error': str(exc)})
            else:
                summary['manifest_errors'].append(relative + ': absent from manifest')
        summary['splits'][split] = {'images': len(images), 'labels': len(labels), 'class_rows': class_counts, 'invalid_label_rows': len(bad)}
        summary['invalid_labels'].extend(bad)
    for name in sorted(set(by_image) - observed_images):
        summary['manifest_errors'].append(name + ': listed image is absent')
    summary['events_crossing_splits'] = [event for event, splits in event_splits.items() if len(splits) > 1]
    summary['mission_groups_crossing_splits'] = [group for group, splits in group_splits.items() if len(splits) > 1]
    summary['image_sha256_count'] = len(seen)
    summary['expected_counts_match'] = (len(manifest) == 1000 and all(
        summary['splits'][split]['images'] == count and summary['splits'][split]['labels'] == count
        for split, count in [('train', 700), ('val', 150), ('test', 150)]))
    summary['valid'] = (summary['expected_counts_match'] and not any(summary[key] for key in
                       ('missing_pairs', 'invalid_labels', 'duplicate_image_sha256',
                        'manifest_errors', 'unreadable_images', 'events_crossing_splits', 'mission_groups_crossing_splits')))
    (out / 'dataset_audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
