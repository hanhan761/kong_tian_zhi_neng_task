"""UAVid data preparation and sequence-separated datasets."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import os
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter

PROJECT = Path('/data/aerospace-intelligence')
DATA = PROJECT / 'data/semantic-segmentation/course-source/datasets/data'
CACHE = PROJECT / 'cache/uavid-512x1024-v1'
DEV_SEQS = frozenset(('seq4', 'seq9', 'seq14', 'seq35'))
CLASS_NAMES = ('clutter', 'building', 'road', 'static_car', 'tree', 'vegetation', 'human', 'moving_car')
PALETTE = ((0,0,0), (128,0,0), (128,64,128), (192,0,192), (0,128,0), (128,128,0), (64,64,0), (64,0,128))
MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


def atomic_save(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as stream:
        np.save(stream, array, allow_pickle=False)
    os.replace(tmp, path)


def record_pairs():
    records = []
    for split in ('train', 'valid'):
        for im in sorted((DATA / split).glob('seq*/Images/*.png')):
            seq = im.parent.parent.name
            mask = DATA / 'labelimg' / split / seq / 'TrainId' / im.name
            if not mask.is_file():
                raise FileNotFoundError(str(mask))
            key = split + '_' + seq + '_' + im.stem
            records.append(dict(key=key, split=split, seq=seq, image=str(im), mask=str(mask)))
    if sum(x['split'] == 'train' for x in records) != 200 or sum(x['split'] == 'valid' for x in records) != 70:
        raise ValueError('unexpected image counts')
    if set(x['seq'] for x in records if x['split'] == 'train').intersection(DEV_SEQS) != DEV_SEQS:
        raise ValueError('development sequences are missing')
    return records


def prepare_one(record):
    image_out = CACHE / 'images' / (record['key'] + '.npy')
    mask_out = CACHE / 'masks' / (record['key'] + '.npy')
    if image_out.is_file() and mask_out.is_file():
        img = np.load(image_out, mmap_mode='r', allow_pickle=False)
        lab = np.load(mask_out, mmap_mode='r', allow_pickle=False)
        if img.shape == (512, 1024, 3) and lab.shape == (512, 1024) and img.dtype == np.uint8 and lab.dtype == np.uint8:
            return
        raise ValueError('invalid existing cache: ' + record['key'])
    with Image.open(record['image']) as im:
        rgb = np.asarray(im.convert('RGB').resize((1024, 512), Image.Resampling.BILINEAR))
    with Image.open(record['mask']) as im:
        mask = np.asarray(im.resize((1024, 512), Image.Resampling.NEAREST))
    if mask.ndim != 2 or not set(np.unique(mask)).issubset(set(range(8)) | {255}):
        raise ValueError('invalid class IDs in ' + record['key'])
    atomic_save(image_out, rgb)
    atomic_save(mask_out, mask)


def prepare():
    records = record_pairs()
    CACHE.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i, _ in enumerate(pool.map(prepare_one, records), 1):
            if i % 25 == 0 or i == len(records):
                print('prepared_images=%d/%d' % (i, len(records)), flush=True)
    groups = {'train': [], 'dev': [], 'official_val': []}
    for record in records:
        group = 'official_val' if record['split'] == 'valid' else ('dev' if record['seq'] in DEV_SEQS else 'train')
        groups[group].append(record)
    if [len(groups[x]) for x in ('train', 'dev', 'official_val')] != [160, 40, 70]:
        raise ValueError('unexpected sequence partition')
    payload = {'version': 1, 'size': [512,1024], 'dev_sequences': sorted(DEV_SEQS), 'classes': CLASS_NAMES, 'groups': groups}
    temp = CACHE / 'manifest.json.tmp'
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, CACHE / 'manifest.json')
    print(json.dumps({'groups': {k: len(v) for k,v in groups.items()}, 'dev_sequences': sorted(DEV_SEQS)}), flush=True)


def load_records(group):
    obj = json.loads((CACHE / 'manifest.json').read_text(encoding='utf-8'))
    if group == 'train_all':
        return obj['groups']['train'] + obj['groups']['dev']
    return obj['groups'][group]


class UAVidImages(Dataset):
    def __init__(self, group, augment=False, full_labels=False):
        self.records = load_records(group)
        self.augment = augment
        self.full_labels = full_labels
        self.jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        img = np.load(CACHE / 'images' / (r['key'] + '.npy'), allow_pickle=False)
        mask = np.load(CACHE / 'masks' / (r['key'] + '.npy'), allow_pickle=False)
        img = Image.fromarray(img)
        if self.augment and random.random() < 0.5:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = np.ascontiguousarray(mask[:, ::-1])
        if self.augment and random.random() < 0.5:
            img = self.jitter(img)
        x = torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2,0,1).float().div_(255)
        x = (x - MEAN) / STD
        if self.full_labels:
            with Image.open(r['mask']) as im:
                mask = np.asarray(im).copy()
        return x, torch.from_numpy(mask.astype(np.int64)), r['key']

if __name__ == '__main__':
    prepare()
