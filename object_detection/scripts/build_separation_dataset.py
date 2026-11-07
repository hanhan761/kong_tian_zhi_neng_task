"""Export a traceable 1,000-frame separation dataset from public NASA videos.

The boxes are deliberately marked as ``bootstrap`` annotations: they are
conservative region boxes generated from reviewed event plans, and must be
opened in LabelImg for a final human pass before claiming strict manual-label
credit. Every image keeps its video alias, timestamp, frame number and source
hash in manifest.json.
"""
import hashlib
import json
import math
import os
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT.parent / '02_原始资料' / 'NASA分离与火工公开影像'
OUT_ROOT = ROOT / 'separation_dataset_1000'

# Each event contributes 50 frames. A video remains entirely in one split.
# Boxes are normalized xyxy coordinates and were chosen after inspecting the
# generated contact sheets. flash segments are the visibly bright plume/smoke
# portions; component is the physical vehicle/debris region to be reviewed.
EVENTS = [
    dict(alias='sts115_srb', split='train', segments=[(0, 18), (20, 31), (32, 47)], component=(.28,.18,.73,.77), flash=(.10,.04,.92,.67), flash_segments=[(0,31)]),
    dict(alias='sts117_et', split='train', segments=[(0, 28), (39, 48)], component=(.20,.15,.78,.88), flash=(.04,.03,.94,.75), flash_segments=[(0,28)]),
    dict(alias='sts117_srb', split='train', segments=[(0, 14), (15, 27), (29, 38), (40, 44)], component=(.24,.16,.76,.82), flash=(.04,.03,.96,.72), flash_segments=[(0,27)]),
    dict(alias='sts119_et', split='train', segments=[(0, 7), (8, 21)], component=(.22,.20,.78,.90), flash=(.08,.05,.95,.82), flash_segments=[(0,21)]),
    dict(alias='sts120_et', split='train', segments=[(0, 35)], component=(.18,.18,.82,.85), flash=(.05,.03,.96,.72), flash_segments=[(20,35)]),
    dict(alias='sts120_srb', split='train', segments=[(0, 12), (16, 36), (47, 61), (64, 74)], component=(.22,.15,.78,.82), flash=(.04,.03,.96,.77), flash_segments=[(0,12),(47,61)]),
    dict(alias='sts121_et', split='train', segments=[(0, 20), (25, 55), (60, 90)], component=(.15,.15,.85,.90), flash=(.03,.02,.97,.82), flash_segments=[(0,20)]),
    dict(alias='sts121_srb', split='train', segments=[(0, 8), (9, 18), (19, 39)], component=(.25,.13,.77,.80), flash=(.05,.02,.96,.70), flash_segments=[(0,18)]),
    dict(alias='sts122_et', split='train', segments=[(0, 18), (20, 43)], component=(.18,.15,.82,.88), flash=(.03,.03,.97,.75), flash_segments=[(0,18)]),
    dict(alias='sts126_et', split='train', segments=[(0, 22), (23, 50)], component=(.18,.15,.82,.88), flash=(.03,.03,.97,.76), flash_segments=[(0,22)]),
    dict(alias='sts128_et', split='train', segments=[(0, 14), (15, 31)], component=(.18,.15,.82,.88), flash=(.03,.03,.97,.76), flash_segments=[(0,14)]),
    dict(alias='sts129_et', split='train', segments=[(0, 18), (19, 41)], component=(.18,.15,.82,.88), flash=(.03,.03,.97,.76), flash_segments=[(0,18)]),
    dict(alias='sts132_et', split='train', segments=[(0, 12), (13, 25)], component=(.18,.15,.82,.88), flash=(.03,.03,.97,.76), flash_segments=[(0,12)]),
    dict(alias='sts133_et', split='train', segments=[(0, 15), (15, 36), (36, 40)], component=(.18,.12,.82,.88), flash=(.05,.03,.96,.83), flash_segments=[(15,36)]),
    dict(alias='sts134_et', split='val', segments=[(0, 18), (18, 43)], component=(.18,.15,.82,.88), flash=(.03,.03,.97,.78), flash_segments=[(0,18)]),
    dict(alias='artemis1_srb', split='val', segments=[(0, 25), (25, 81)], component=(.15,.12,.85,.88), flash=(.03,.03,.97,.80), flash_segments=[(0,25)]),
    dict(alias='artemis1_icps', split='val', segments=[(20, 42), (42, 70)], component=(.18,.15,.82,.86), flash=(.04,.03,.96,.75), flash_segments=[(20,42)]),
    dict(alias='orion_abort', split='test', segments=[(8, 35), (35, 90)], component=(.12,.10,.88,.90), flash=(.03,.03,.97,.78), flash_segments=[(8,35)]),
    dict(alias='artemis2_fairing', split='test', segments=[(3, 10), (10, 48)], component=(.08,.08,.92,.92), flash=(.02,.02,.98,.75), flash_segments=[(3,10)]),
    dict(alias='srb_ground_test', split='test', segments=[(43, 64)], component=(.15,.12,.85,.90), flash=(.03,.03,.97,.80), flash_segments=[(43,64)]),
]


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def in_segments(t, segments):
    return any(a <= t <= b for a, b in segments)


def sample_times(segments, count):
    lengths = [max(0.01, b - a) for a, b in segments]
    total = sum(lengths)
    allocations = [max(1, round(count * x / total)) for x in lengths]
    while sum(allocations) > count:
        i = max(range(len(allocations)), key=lambda j: allocations[j])
        if allocations[i] > 1:
            allocations[i] -= 1
        else:
            break
    while sum(allocations) < count:
        allocations[sum(allocations) % len(allocations)] += 1
    times = []
    for (a, b), n in zip(segments, allocations):
        if n == 1:
            times.append((a + b) / 2)
        else:
            times.extend(a + (b - a) * i / (n - 1) for i in range(n))
    return times[:count]


def yolo_box(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)


def main():
    if len(EVENTS) * 50 != 1000:
        raise RuntimeError('Event plan must contain exactly 20 events')
    for split in ('train', 'val', 'test'):
        (OUT_ROOT / 'images' / split).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / 'labels' / split).mkdir(parents=True, exist_ok=True)
    manifest = []
    counts = {k: 0 for k in ('train', 'val', 'test')}
    seen_image_hashes = set()
    for event in EVENTS:
        video_dir = SOURCE_ROOT / event['alias']
        video = video_dir / 'source_original.mp4'
        if not video.exists():
            video = video_dir / 'source.mp4'
        if not video.exists():
            raise FileNotFoundError(video)
        cap = cv2.VideoCapture(str(video))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 1.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_sha = sha256(video)
        times = sample_times(event['segments'], 200)
        written = 0
        for seconds in times:
            if written >= 50:
                break
            frame_no = max(0, int(round(seconds * fps)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f'Could not read {event["alias"]} frame {frame_no}')
            # Exclude black leader/trailer frames that contain no observable
            # event; these are common in the public edits.
            if float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()) < 5.0:
                continue
            actual_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            actual_seconds = actual_frame / fps
            index = written
            stem = f'{event["alias"]}_{index:03d}'
            image_path = OUT_ROOT / 'images' / event['split'] / f'{stem}.jpg'
            label_path = OUT_ROOT / 'labels' / event['split'] / f'{stem}.txt'
            # cv2.imwrite cannot reliably open a Windows path containing Chinese
            # characters. Encode in memory, then let pathlib write the bytes.
            ok_jpg, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if not ok_jpg:
                raise RuntimeError(f'Could not encode {event["alias"]} frame {frame_no}')
            encoded_bytes = encoded.tobytes()
            image_hash = hashlib.sha256(encoded_bytes).hexdigest()
            if image_hash in seen_image_hashes:
                continue
            seen_image_hashes.add(image_hash)
            image_path = OUT_ROOT / 'images' / event['split'] / f'{stem}.jpg'
            image_path.write_bytes(encoded_bytes)
            rows = [f'0 {" ".join(f"{v:.8f}" for v in yolo_box(event["component"]))}']
            if in_segments(actual_seconds, event['flash_segments']):
                rows.append(f'1 {" ".join(f"{v:.8f}" for v in yolo_box(event["flash"]))}')
            label_path.write_text('\n'.join(rows) + '\n', encoding='utf-8')
            manifest.append(dict(image=str(image_path.relative_to(OUT_ROOT)), label=str(label_path.relative_to(OUT_ROOT)),
                                 alias=event['alias'], split=event['split'], source_video=str(video),
                                 source_sha256=source_sha, frame=actual_frame, timestamp_seconds=actual_seconds,
                                 width=width, height=height, annotation_method='bootstrap_region_box_needs_LabelImg_review',
                                 classes=['separation_component'] + (['flash_smoke'] if len(rows) == 2 else [])))
            counts[event['split']] += 1
            written += 1
        if written != 50:
            raise RuntimeError(f'Only {written} unique frames available for {event["alias"]}')
        cap.release()
    (OUT_ROOT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    yaml_text = '''# Public NASA video frame dataset for separation-event research.\n# Images are grouped by source event; no source video crosses splits.\npath: separation_dataset_1000\ntrain: images/train\nval: images/val\ntest: images/test\nnc: 2\nnames: [separation_component, flash_smoke]\n'''
    (OUT_ROOT / 'separation_dataset.yaml').write_text(yaml_text, encoding='utf-8')
    test_yaml = yaml_text.replace('val: images/val', 'val: images/test')
    (OUT_ROOT / 'separation_dataset_test.yaml').write_text(test_yaml, encoding='utf-8')
    summary = dict(total=len(manifest), splits=counts, classes={'separation_component': sum('separation_component' in x['classes'] for x in manifest), 'flash_smoke': sum('flash_smoke' in x['classes'] for x in manifest)}, events=[x['alias'] for x in EVENTS], annotation_method='bootstrap_region_box_needs_LabelImg_review')
    (OUT_ROOT / 'dataset_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
