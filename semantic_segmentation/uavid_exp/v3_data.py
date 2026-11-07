"""Sequence-disjoint nested splits and bounded expert hard-negative sampling."""
from itertools import combinations
import json
import random

import numpy as np
from PIL import Image, ImageEnhance
import torch

from v2_data import INDEX, ROOT, RARE, TrainingImages, tensor_image


def nested_split(fold):
    obj = json.loads(INDEX.read_text(encoding='utf-8'))
    outer_seqs = set(obj['folds'][fold])
    items = obj['items']
    outer_train = [r for r in items if r['seq'] not in outer_seqs]
    outer_val = [r for r in items if r['seq'] in outer_seqs]
    seqs = sorted({r['seq'] for r in outer_train})
    stats = {}
    for seq in seqs:
        rows = [r for r in outer_train if r['seq'] == seq]
        stats[seq] = np.asarray([sum(r['counts'][str(c)] for r in rows) for c in RARE]
                                + [sum(r['counts']['6'] > 0 for r in rows)], dtype=np.float64)
    total = np.maximum(sum(stats.values()), 1.)
    target = 4. / len(seqs)
    def objective(group):
        v = sum((stats[s] for s in group), np.zeros(4)) / total
        return float(np.square(v-target).sum())
    inner_seqs = set(min(combinations(seqs, 4), key=lambda group: (objective(group), group)))
    inner_train = [r for r in outer_train if r['seq'] not in inner_seqs]
    inner_val = [r for r in outer_train if r['seq'] in inner_seqs]
    sets = [set(r['seq'] for r in z) for z in (inner_train, inner_val, outer_val)]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i+1, 3)):
        raise RuntimeError('nested sequence leakage')
    if not all(any(r['counts']['6'] for r in part) for part in (inner_train, inner_val, outer_val)):
        raise RuntimeError('nested split lacks human labels')
    return {'inner_train': inner_train, 'inner_val': inner_val,
            'outer_train': outer_train, 'outer_val': outer_val}


class HardNegativeImages(TrainingImages):
    def __init__(self, records, mode, rare_fraction=0., expert=False, samples=300, hard_path=None):
        super().__init__(records, mode, rare_fraction, expert, samples)
        self.hard = []
        if hard_path:
            if not expert or mode != 'tile':
                raise ValueError('hard negatives require tiled human expert')
            index = {r['key']: r for r in records}
            obj = json.loads(open(hard_path, encoding='utf-8').read())
            for entry in obj['entries']:
                if entry['key'] not in index:
                    raise ValueError('hard negative does not belong to training split')
                if entry['coords']:
                    self.hard.append((index[entry['key']], entry['coords']))

    def __getitem__(self, index):
        if not self.hard or random.random() >= .25:
            return super().__getitem__(index)
        record, coords = random.choice(self.hard)
        image = np.load(ROOT/'images'/(record['key']+'.npy'), mmap_mode='r', allow_pickle=False)
        mask = np.load(ROOT/'masks'/(record['key']+'.npy'), mmap_mode='r', allow_pickle=False)
        h, w = mask.shape
        cy, cx = divmod(random.choice(coords), w)
        side = 512
        y = max(0, min(cy-side//2+random.randint(-128, 128), h-side))
        x = max(0, min(cx-side//2+random.randint(-128, 128), w-side))
        image = np.array(image[y:y+side, x:x+side], copy=True)
        mask = np.array(mask[y:y+side, x:x+side], copy=True)
        if random.random() < .5:
            image, mask = np.ascontiguousarray(image[:, ::-1]), np.ascontiguousarray(mask[:, ::-1])
        pil = Image.fromarray(image)
        if random.random() < .5:
            pil = ImageEnhance.Brightness(pil).enhance(random.uniform(.8, 1.2))
            pil = ImageEnhance.Color(pil).enhance(random.uniform(.8, 1.2))
        mask = np.where(mask == 255, 255, (mask == 6).astype(np.uint8))
        return tensor_image(pil), torch.from_numpy(mask.astype(np.int64))
