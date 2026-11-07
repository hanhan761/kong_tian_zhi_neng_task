"""Package the completed assisted-label workflow; preserve every v2 image/label byte."""
from collections import Counter
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'separation_dataset_v2_1000'
DEST = ROOT / 'separation_dataset_ready1000'
OUT = ROOT / 'reports/06_新一轮优化_20261023/数据集完成交付'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def previews(manifest):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 17)
    selected = []
    for alias in sorted({x['alias'] for x in manifest}):
        rows = [x for x in manifest if x['alias'] == alias]
        selected.extend([rows[0], rows[len(rows) // 2], rows[-1]])
    cell_w, cell_h = 400, 280
    canvas = Image.new('RGB', (cell_w * 3, cell_h * 12), '#111827')
    for i, item in enumerate(selected):
        image = Image.open(DEST / item['image']).convert('RGB')
        draw = ImageDraw.Draw(image)
        w, h = image.size
        for line in (DEST / item['label']).read_text(encoding='utf-8').splitlines():
            cls, cx, cy, bw, bh = map(float, line.split())
            box = [(cx - bw/2)*w, (cy - bh/2)*h, (cx + bw/2)*w, (cy + bh/2)*h]
            color = '#22ff77' if cls == 0 else '#ffb347'
            draw.rectangle(box, outline=color, width=2)
            draw.text((box[0], max(0, box[1]-21)), str(int(cls)), font=font, fill=color)
        image.thumbnail((390, 235))
        x, y = (i % 3) * cell_w, (i // 3) * cell_h
        canvas.paste(image, (x + (cell_w-image.width)//2, y+40))
        ImageDraw.Draw(canvas).text((x+5, y+3), Path(item['image']).name, font=font, fill='white')
        ImageDraw.Draw(canvas).text((x+5, y+23), item['split']+' / '+str(round(item['timestamp_seconds'], 2))+'s', font=font, fill='#b3c6ef')
    canvas.save(DEST / '标签抽样检查_12片段首中末.jpg', quality=92)
    write_json(DEST / 'preview_samples.json', [dict(image=x['image'], label=x['label']) for x in selected])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-partial', action='store_true')
    args = parser.parse_args()
    archive = ROOT.parent / '04_可直接训练的数据集_1000张.zip'
    if (OUT / 'export_record.json').exists() or archive.exists():
        raise FileExistsError('Completed delivery already exists.')
    if (DEST.exists() or OUT.exists()) and not args.resume_partial:
        raise FileExistsError('Completed export already exists. Preserve it; use a new version for changes.')
    manifest = json.loads((SOURCE / 'manifest.json').read_text(encoding='utf-8'))
    original = json.loads((SOURCE / 'dataset_summary.json').read_text(encoding='utf-8'))
    DEST.mkdir(exist_ok=args.resume_partial)
    OUT.mkdir(parents=True, exist_ok=args.resume_partial)
    for item in manifest:
        for key in ('image', 'label'):
            relative = item[key]
            source, target = SOURCE / relative, DEST / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and digest(target) != item[key + '_sha256']:
                raise ValueError('Existing delivery file changed: ' + relative)
            shutil.copy2(source, target)
            if digest(target) != item[key + '_sha256']:
                raise ValueError('Changed bytes: ' + relative)
        # Keep the original review flag false. This release completes the
        # assisted workflow, not a fictitious 1000-image manual review.
        item['source_video'] = Path(item['source_video']).relative_to(ROOT.parent).as_posix()
        item['delivery_workflow'] = 'completed_keyframe_assisted_labels'
    write_json(DEST / 'manifest.json', manifest)
    for name in ('annotation_plan.json', 'classes.txt'):
        shutil.copy2(SOURCE / name, DEST / name)
    original.update(version='ready1000_assisted_release',
                    annotation_workflow_complete=True, manual_LabelImg_required=False,
                    image_label_bytes_identical_to='separation_dataset_v2_1000',
                    note='Completed assisted-label release, as requested. No per-image human LabelImg review is claimed. Semantic errors can remain; see dataset_card.md.')
    write_json(DEST / 'dataset_summary.json', original)
    source_root = ROOT.parent / '02_原始资料/NASA分离与火工公开影像'
    sources = []
    for alias in sorted({m['alias'] for m in manifest}):
        folder = source_root / alias
        source = json.loads((folder / 'provenance.json').read_text(encoding='utf-8'))
        original_path = folder / 'original_provenance.json'
        original_source = json.loads(original_path.read_text(encoding='utf-8')) if original_path.exists() else dict(source)
        actual_hash = next(m['source_sha256'] for m in manifest if m['alias'] == alias)
        if original_source['sha256'] != actual_hash:
            raise ValueError('Source metadata hash mismatch: ' + alias)
        source['frame_source'] = original_source
        source['dataset_alias'] = alias
        sources.append(source)
    write_json(DEST / 'sources.json', sources)
    for suffix, validation in [('', 'images/val'), ('_test', 'images/test')]:
        (DEST / ('separation_dataset' + suffix + '.yaml')).write_text(
            '# Assisted labels; see dataset_card.md. Resolve path from the YOLO project root.\n'
            'path: separation_dataset_ready1000\ntrain: images/train\nval: '+validation+
            '\ntest: images/test\nnc: 2\nnames: [separation_component, flash_smoke]\n', encoding='utf-8')
    (DEST / 'configure_paths.py').write_text('''"""Run after moving this dataset: python configure_paths.py"""
from pathlib import Path
import json
root = Path(__file__).resolve().parent
for suffix, split in [('', 'val'), ('_test', 'test')]:
    text = 'path: '+json.dumps(root.as_posix(), ensure_ascii=False)+'\\n'
    text += 'train: images/train\\nval: images/'+split+'\\ntest: images/test\\n'
    text += 'nc: 2\\nnames: [separation_component, flash_smoke]\\n'
    (root / ('runtime'+suffix+'.yaml')).write_text(text, encoding='utf-8')
    print(root / ('runtime'+suffix+'.yaml'))
''', encoding='utf-8')
    card = ROOT / 'data/separation_ready_dataset_card.md'
    shutil.copy2(card, DEST / 'dataset_card.md')
    review = ROOT / 'reports/03_专业数据集核查/v2_keyframe_assisted/视频语义审查与修订说明.md'
    shutil.copy2(review, DEST / '辅助标注语义修订记录.md')
    previews(manifest)
    checks = {p.relative_to(DEST).as_posix(): digest(p) for p in sorted(DEST.rglob('*')) if p.is_file()}
    write_json(DEST / 'checksums.json', checks)
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(DEST.rglob('*')):
            if path.is_file():
                z.write(path, path.relative_to(ROOT).as_posix())
    record = dict(created=datetime.now().astimezone().isoformat(), dataset=str(DEST),
                  archive=str(archive), archive_sha256=digest(archive), archive_bytes=archive.stat().st_size,
                  images=len(manifest), labels=len(manifest), split_counts=dict(Counter(x['split'] for x in manifest)),
                  all_1000_image_and_label_hashes_identical_to_v2=True,
                  annotation_workflow_complete=True, manual_LabelImg_required=False,
                  all_human_LabelImg_reviewed_flags_false=all(not x['human_LabelImg_reviewed'] for x in manifest),
                  annotation_method=original['annotation_method'],
                  meaning='Completed assisted dataset; no claim of fully manual ground truth or perfect semantic accuracy.')
    write_json(OUT / 'export_record.json', record)
    print(json.dumps(record, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
