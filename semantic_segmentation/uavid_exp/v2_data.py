"""Native-resolution UAVid cache and deterministic sequence-separated folds."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import os
import random
import zlib

import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset

from data import CACHE, MEAN, STD, atomic_save, load_records

ROOT = Path('/data/aerospace-intelligence/cache/uavid-native-v2')
INDEX = ROOT / 'index.json'
RARE = (6, 3, 7)


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def cache_one(record):
    key = record['key']
    image_path = ROOT / 'images' / (key + '.npy')
    mask_path = ROOT / 'masks' / (key + '.npy')
    with Image.open(record['mask']) as im:
        mask = np.asarray(im).copy()
    if mask.ndim != 2 or not set(np.unique(mask)).issubset(set(range(8)) | {255}):
        raise ValueError('invalid mask: ' + key)
    if not image_path.exists():
        with Image.open(record['image']) as im:
            image = np.asarray(im.convert('RGB'))
        if image.shape[:2] != mask.shape:
            raise ValueError('image and mask shape mismatch: ' + key)
        atomic_save(image_path, image)
    if not mask_path.exists():
        atomic_save(mask_path, mask)
    if np.load(image_path, mmap_mode='r').shape != (*mask.shape, 3):
        raise ValueError('invalid cached image: ' + key)
    if np.load(mask_path, mmap_mode='r').shape != mask.shape:
        raise ValueError('invalid cached mask: ' + key)
    rng = np.random.default_rng(zlib.crc32(key.encode('utf-8')))
    coords, counts = {}, {}
    for cls in RARE:
        flat = np.flatnonzero(mask == cls)
        counts[str(cls)] = int(flat.size)
        if flat.size > 4096:
            flat = rng.choice(flat, 4096, replace=False)
        coords[str(cls)] = flat.astype(np.int64).tolist()
    return {'key': key, 'seq': record['seq'], 'height': mask.shape[0],
            'width': mask.shape[1], 'counts': counts, 'coords': coords}


def assign_folds(items):
    seqs = sorted({r['seq'] for r in items})
    if len(seqs) != 20:
        raise ValueError('expected 20 training sequences')
    # Balance pixel totals and number of images with humans, keeping 7/7/6 sequences.
    stats = {}
    for seq in seqs:
        rows = [r for r in items if r['seq'] == seq]
        stats[seq] = np.array([sum(r['counts'][str(c)] for r in rows) for c in RARE]
                              + [sum(r['counts']['6'] > 0 for r in rows)], dtype=np.float64)
    order = sorted(seqs, key=lambda s: (-stats[s][0], s))
    groups = [[], [], []]
    for i, seq in enumerate(order):
        block, offset = divmod(i, 3)
        groups[offset if block % 2 == 0 else 2 - offset].append(seq)
    denom = np.maximum(sum(stats.values()), 1.0)
    def objective(g):
        shares = np.stack([sum((stats[s] for s in a), np.zeros(4)) / denom for a in g])
        return float(np.square(shares - 1/3).sum())
    rng = random.Random(42)
    best = objective(groups)
    for _ in range(3000):
        a, b = rng.sample(range(3), 2)
        ia, ib = rng.randrange(len(groups[a])), rng.randrange(len(groups[b]))
        groups[a][ia], groups[b][ib] = groups[b][ib], groups[a][ia]
        score = objective(groups)
        if score < best:
            best = score
        else:
            groups[a][ia], groups[b][ib] = groups[b][ib], groups[a][ia]
    return [sorted(g) for g in groups]


def prepare():
    if INDEX.exists():
        data = json.loads(INDEX.read_text(encoding='utf-8'))
        if data.get('version') != 2 or len(data['items']) != 200:
            raise ValueError('native index version/count mismatch')
        return data
    records = load_records('train_all')
    if len(records) != 200:
        raise ValueError('expected 200 training images')
    ROOT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        items = []
        for i, obj in enumerate(pool.map(cache_one, records), 1):
            items.append(obj)
            if i % 25 == 0 or i == len(records):
                print(json.dumps({'stage': 'native_cache', 'images': i, 'total': len(records)}), flush=True)
    folds = assign_folds(items)
    if sorted(sum(folds, [])) != sorted({r['seq'] for r in items}):
        raise RuntimeError('fold overlap or omission')
    data = {'version': 2, 'seed': 42, 'folds': folds, 'items': items}
    atomic_json(INDEX, data)
    return data


def get_split(fold, train_all=False):
    data = json.loads(INDEX.read_text(encoding='utf-8'))
    if train_all:
        return data['items'], []
    heldout = set(data['folds'][fold])
    train = [r for r in data['items'] if r['seq'] not in heldout]
    dev = [r for r in data['items'] if r['seq'] in heldout]
    if set(r['seq'] for r in train) & set(r['seq'] for r in dev):
        raise RuntimeError('sequence leakage')
    return train, dev


def tensor_image(image):
    x = torch.from_numpy(np.array(image, dtype=np.uint8)).permute(2, 0, 1).float().div_(255)
    return (x - MEAN) / STD


class TrainingImages(Dataset):
    def __init__(self, records, mode, rare_fraction=0.0, expert=False, samples=300):
        self.records = records
        self.mode, self.rare_fraction, self.expert, self.samples = mode, rare_fraction, expert, samples
        self.by_class = {c: [r for r in records if r['counts'][str(c)] > 0] for c in RARE}
        if not records or (rare_fraction and not self.by_class[6]):
            raise ValueError('training split lacks required samples')

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        record = random.choice(self.records)
        cls = None
        if self.mode == 'tile' and random.random() < self.rare_fraction:
            cls = 6 if self.expert else random.choices(RARE, weights=(2, 1, 1), k=1)[0]
            if self.by_class[cls]:
                record = random.choice(self.by_class[cls])
            else:
                cls = None
        if self.mode == 'global':
            image = np.load(CACHE / 'images' / (record['key'] + '.npy'), allow_pickle=False)
            mask = np.load(CACHE / 'masks' / (record['key'] + '.npy'), allow_pickle=False)
        else:
            side = 512 if self.expert else 1024
            image = np.load(ROOT / 'images' / (record['key'] + '.npy'), mmap_mode='r', allow_pickle=False)
            mask = np.load(ROOT / 'masks' / (record['key'] + '.npy'), mmap_mode='r', allow_pickle=False)
            h, w = mask.shape
            if cls is not None:
                flat = random.choice(record['coords'][str(cls)])
                cy, cx = divmod(flat, w)
                y = cy - side // 2 + random.randint(-side // 4, side // 4)
                x = cx - side // 2 + random.randint(-side // 4, side // 4)
                y, x = max(0, min(y, h-side)), max(0, min(x, w-side))
            else:
                y, x = random.randint(0, max(0, h-side)), random.randint(0, max(0, w-side))
            image = np.array(image[y:y+side, x:x+side], copy=True)
            mask = np.array(mask[y:y+side, x:x+side], copy=True)
        if random.random() < 0.5:
            image, mask = np.ascontiguousarray(image[:, ::-1]), np.ascontiguousarray(mask[:, ::-1])
        pil = Image.fromarray(image)
        if random.random() < 0.5:
            pil = ImageEnhance.Brightness(pil).enhance(random.uniform(0.8, 1.2))
            pil = ImageEnhance.Color(pil).enhance(random.uniform(0.8, 1.2))
        if self.expert:
            mask = np.where(mask == 255, 255, (mask == 6).astype(np.uint8))
        return tensor_image(pil), torch.from_numpy(mask.astype(np.int64))


if __name__ == '__main__':
    obj = prepare()
    print(json.dumps({'folds': obj['folds'], 'images': len(obj['items'])}), flush=True)
