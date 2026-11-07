"""Build a new, keyframe-assisted dataset without touching the bootstrap version.

The plan is versioned JSON. Boxes are drawn from visually inspected keyframes,
then interpolated within each continuous shot. This is assisted annotation,
not a claim that a person annotated all images in LabelImg.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import zipfile
from datetime import datetime

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / '02_原始资料/NASA分离与火工公开影像'
DATA = ROOT / 'separation_dataset_v2_1000'
REPORT = ROOT / 'reports/03_专业数据集核查/v2_keyframe_assisted'
PLAN = ROOT / 'data/separation_v2_plan.json'
COLORS = [(40, 220, 40), (0, 140, 255)]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def save_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, array = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError(path)
    path.write_bytes(array.tobytes())


def video_path(alias):
    folder = SOURCE / alias
    path = folder / 'source_original.mp4'
    return path if path.exists() else folder / 'source.mp4'


def read_frame(cap, seconds):
    fps = cap.get(cv2.CAP_PROP_FPS)
    number = int(round(seconds * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, number)
    ok, image = cap.read()
    if not ok:
        raise RuntimeError('Cannot decode frame {}'.format(number))
    return image, number, number / fps


def sheet(items, path, columns=3):
    tiles = []
    for caption, image in items:
        tile = np.zeros((256, 480, 3), np.uint8)
        h, w = image.shape[:2]
        scale = min(480 / w, 226 / h)
        resized = cv2.resize(image, (int(w * scale), int(h * scale)))
        tile[30:30 + resized.shape[0], :resized.shape[1]] = resized
        cv2.putText(tile, caption, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))
    save_image(path, np.vstack([np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)]))


def interpolate(track, seconds):
    keys = track['keyframes']
    if seconds < keys[0][0] - .03 or seconds > keys[-1][0] + .03:
        return None
    for left, right in zip(keys, keys[1:]):
        if left[0] <= seconds <= right[0]:
            fraction = (seconds - left[0]) / (right[0] - left[0])
            return [a + fraction * (b - a) for a, b in zip(left[1:], right[1:])]
    return keys[0][1:] if seconds <= keys[0][0] else keys[-1][1:]


def refine_tank(image, box):
    """Tighten the orange tank in each frame; cloud background is unsaturated.

    This image-based correction is part of annotation, never model evaluation.
    Keep the interpolated keyframe if no plausible orange component is found.
    """
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(image.astype(np.float32))
    mask = ((hsv[:, :, 0] < 28) & (hsv[:, :, 1] > 50) & (r > g * 1.16) & (r > b * 1.22)).astype(np.uint8)
    roi = np.zeros_like(mask)
    roi[int(.08 * h):int(.75 * h), int(.24 * w):int(.75 * w)] = 1
    mask *= roi
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 7), np.uint8))
    n, components, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    eligible = [i for i in range(1, n) if 12 <= stats[i, cv2.CC_STAT_AREA] <= w * h * .03]
    if not eligible:
        return box
    best = max(eligible, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    x, y, bw, bh, area = stats[best]
    # Compression/interlacing can break one orange tank into several pieces.
    nearby = [i for i in eligible if abs(centroids[i][0] - centroids[best][0]) < .07 * w
              and abs(centroids[i][1] - centroids[best][1]) < .10 * h]
    x2, y2 = x + bw, y + bh
    for index in nearby:
        sx, sy, sw, sh, _ = stats[index]
        x, y, x2, y2 = min(x, sx), min(y, sy), max(x2, sx + sw), max(y2, sy + sh)
    bw, bh = x2 - x, y2 - y
    return [(x - 5) / w, (y - 4) / h, (x + bw + 5) / w, (y + bh + 4) / h]


def white_core_shift(image, expected):
    """Measure common camera jitter from the bright core in STS117's blue sky.

    Only this reviewed continuous shot uses this correction. It is an annotation
    aid, not a learned detector or an evaluation-time modification.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = expected
    left, right = max(0, int((x1 - .018) * w)), min(w, int((x2 + .018) * w))
    top, bottom = max(0, int((y1 - .11) * h)), min(h, int((y2 + .11) * h))
    region = image[top:bottom, left:right].astype(np.float32)
    b, g, r = cv2.split(region)
    mask = ((r > 115) & (r - .65 * b > 18)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    candidates = [i for i in range(1, n) if 15 <= stats[i, 4] <= w * h * .003]
    if not candidates:
        return 0.0, 0.0, expected
    # In this shot the core is below both separated boosters. Largest-area
    # selection can mistakenly lock onto a booster when the camera jumps.
    best = max(candidates, key=lambda i: centers[i][1])
    x, y, bw, bh, area = stats[best]
    measured = [(left + x - 4) / w, (top + y - 3) / h,
                (left + x + bw + 4) / w, (top + y + bh + 3) / h]
    dx = (measured[0] + measured[2] - x1 - x2) / 2
    dy = (measured[1] + measured[3] - y1 - y2) / 2
    return dx, dy, measured


def preview(plan):
    for event in plan['events']:
        cap = cv2.VideoCapture(str(video_path(event['alias'])))
        items = []
        for seconds in np.linspace(*event['interval'], 12):
            image, number, actual = read_frame(cap, seconds)
            h, w = image.shape[:2]
            for fraction in [.2, .4, .6, .8]:
                cv2.line(image, (int(w * fraction), 0), (int(w * fraction), h), (90, 90, 90), 1)
                cv2.line(image, (0, int(h * fraction)), (w, int(h * fraction)), (90, 90, 90), 1)
            items.append((f"{event['alias']} {actual:.3f}s f{number}", image))
        cap.release()
        sheet(items, REPORT / 'raw_keyframes' / (event['alias'] + '.jpg'))
    print('Raw keyframe sheets:', REPORT / 'raw_keyframes')


def build(plan, revise=False):
    if DATA.exists() and not revise:
        raise FileExistsError('Dataset exists. Preserve it; choose a new version before rebuilding.')
    if revise:
        if not (DATA / 'manifest.json').is_file():
            raise FileNotFoundError('No existing dataset to revise')
        # Refuse to silently rewrite any image when revising annotations.
        before = json.loads((DATA / 'manifest.json').read_text(encoding='utf-8'))
        if len(before) != 1000:
            raise ValueError('Cannot revise an incomplete dataset')
        for item in before:
            if digest(DATA / item['image']) != item['image_sha256']:
                raise ValueError('Image changed: ' + item['image'])
        archive = REPORT / ('labels_before_revision_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.zip')
        with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
            for item in before:
                z.write(DATA / item['label'], item['label'])
            for name in ['manifest.json', 'annotation_plan.json', 'dataset_summary.json']:
                z.write(DATA / name, name)
    if sum(event['count'] for event in plan['events']) != 1000:
        raise ValueError('Plan must contain exactly 1000 frames')
    DATA.mkdir(exist_ok=revise)
    manifest, seen, groups = [], set(), defaultdict(set)
    for event in plan['events']:
        if not event.get('tracks'):
            raise ValueError('Missing visually specified tracks: ' + event['alias'])
        source = video_path(event['alias'])
        source_hash = digest(source)
        cap = cv2.VideoCapture(str(source))
        items, frames = [], set()
        for index, seconds in enumerate(np.linspace(*event['interval'], event['count'])):
            image, number, actual = read_frame(cap, seconds)
            if number in frames:
                raise ValueError('Frame repeated within event')
            frames.add(number)
            h, w = image.shape[:2]
            stem = '{}_{:03d}'.format(event['alias'], index)
            image_path = DATA / 'images' / event['split'] / (stem + '.jpg')
            label_path = DATA / 'labels' / event['split'] / (stem + '.txt')
            if not revise:
                save_image(image_path, image)
            image_hash = digest(image_path)
            if image_hash in seen:
                raise ValueError('Byte-identical image: ' + stem)
            seen.add(image_hash)
            labels, classes = [], []
            drawn = image.copy()
            shift = None
            if event.get('refine') == 'white_core_camera_shift':
                core = next(track for track in event['tracks'] if track['name'] == 'core_orbiter')
                shift = white_core_shift(image, interpolate(core, actual))
            for track in event['tracks']:
                box = interpolate(track, actual)
                if box is None:
                    continue
                if event.get('refine') == 'orange_tank':
                    box = refine_tank(image, box)
                if shift is not None:
                    dx, dy, measured = shift
                    box = measured if track['name'] == 'core_orbiter' else [
                        box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy]
                x1, y1, x2, y2 = np.clip(box, 0, 1)
                if x2 <= x1 or y2 <= y1:
                    raise ValueError('Invalid box: ' + stem)
                cls = track['class']
                values = [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
                labels.append(str(cls) + ' ' + ' '.join('{:.8f}'.format(v) for v in values))
                classes.append(cls)
                cv2.rectangle(drawn, (round(x1 * w), round(y1 * h)), (round(x2 * w), round(y2 * h)), COLORS[cls], 2)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text('\n'.join(labels) + ('\n' if labels else ''), encoding='utf-8')
            manifest.append(dict(image=image_path.relative_to(DATA).as_posix(),
                                 label=label_path.relative_to(DATA).as_posix(),
                                 image_sha256=image_hash, label_sha256=digest(label_path),
                                 alias=event['alias'], group=event['group'], split=event['split'],
                                 source_video=str(source), source_sha256=source_hash,
                                 frame=number, timestamp_seconds=actual, width=w, height=h,
                                 classes=classes, annotation_method='AI_visual_keyframes_linear_interpolation',
                                 per_image_refinement=event.get('refine'),
                                 human_LabelImg_reviewed=False))
            groups[event['group']].add(event['split'])
            items.append((f'{stem} {actual:.3f}s', drawn))
        cap.release()
        # Every exported image is included, including frames between keyframes.
        for offset in range(0, len(items), 15):
            sheet(items[offset:offset + 15], REPORT / 'all_labels' / ('{}_{:03d}.jpg'.format(event['alias'], offset)))
    if any(len(splits) > 1 for splits in groups.values()):
        raise ValueError('A mission group crosses splits')
    summary = dict(total=len(manifest), splits=dict(Counter(x['split'] for x in manifest)),
                   class_boxes=dict(Counter(c for x in manifest for c in x['classes'])),
                   background_images=sum(not x['classes'] for x in manifest),
                   groups={k: sorted(v) for k, v in groups.items()},
                   annotation_method='AI_visual_keyframes_linear_interpolation',
                   strict_manual_LabelImg_annotation_complete=False,
                   plan_sha256=digest(PLAN),
                   note='Real video frames; adjacent frames are correlated. Per-image human LabelImg refinement remains a separate task.')
    save_json(DATA / 'manifest.json', manifest)
    save_json(DATA / 'dataset_summary.json', summary)
    save_json(DATA / 'annotation_plan.json', plan)
    names = 'separation_component\nflash_smoke\n'
    (DATA / 'classes.txt').write_text(names, encoding='utf-8')
    for split in ['train', 'val', 'test']:
        (DATA / 'labels' / split / 'classes.txt').write_text(names, encoding='utf-8')
    yaml = ('# AI keyframe assisted annotations; see manifest.json.\n'
            'path: separation_dataset_v2_1000\ntrain: images/train\nval: images/val\n'
            'test: images/test\nnc: 2\nnames: [separation_component, flash_smoke]\n')
    (DATA / 'separation_dataset.yaml').write_text(yaml, encoding='utf-8')
    (DATA / 'separation_dataset_test.yaml').write_text(yaml.replace('val: images/val', 'val: images/test'), encoding='utf-8')
    save_json(REPORT / 'dataset_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--revise-labels', action='store_true', help='Archive prior labels, then revise only annotations before a NEW training run')
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    preview(plan) if args.preview else build(plan, args.revise_labels)


if __name__ == '__main__':
    main()
