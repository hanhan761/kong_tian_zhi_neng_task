"""Small-object expert patches from training and development sequences only."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import os
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter
from data import PROJECT, MEAN, STD, load_records, atomic_save

PATCH_CACHE = PROJECT / 'cache/uavid-expert-512-v1'
PATCH = 512


def create_patches(record):
    key = record['key']
    with Image.open(record['image']) as im:
        image = np.asarray(im.convert('RGB'))
    with Image.open(record['mask']) as im:
        mask = np.asarray(im)
    if image.shape[:2] != mask.shape or mask.ndim != 2:
        raise ValueError('bad source shape: ' + key)
    h, w = mask.shape
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little')
    rng = np.random.default_rng(seed)
    coordinates = []
    for target in (6, 7, 3, None):
        indices = np.flatnonzero(mask == target) if target is not None else np.empty(0, dtype=np.int64)
        if len(indices):
            index = int(indices[rng.integers(len(indices))])
            center_y, center_x = divmod(index, w)
        else:
            center_y, center_x = int(rng.integers(h)), int(rng.integers(w))
        x = max(0, min(w-PATCH, center_x-int(rng.integers(PATCH//4, 3*PATCH//4))))
        y = max(0, min(h-PATCH, center_y-int(rng.integers(PATCH//4, 3*PATCH//4))))
        coordinates.append((int(y), int(x)))
    result = []
    for number, (y,x) in enumerate(coordinates):
        patch_key = key + '_p' + str(number)
        ip = PATCH_CACHE/'images'/(patch_key+'.npy')
        mp = PATCH_CACHE/'masks'/(patch_key+'.npy')
        if not ip.exists() or not mp.exists():
            atomic_save(ip, image[y:y+PATCH,x:x+PATCH].copy())
            atomic_save(mp, mask[y:y+PATCH,x:x+PATCH].copy())
        else:
            if np.load(ip,mmap_mode='r',allow_pickle=False).shape != (PATCH,PATCH,3) or np.load(mp,mmap_mode='r',allow_pickle=False).shape != (PATCH,PATCH):
                raise ValueError('invalid patch cache: ' + patch_key)
        result.append({'key':patch_key,'image_key':key,'seq':record['seq'],'y':y,'x':x})
    return result


def prepare():
    PATCH_CACHE.mkdir(parents=True, exist_ok=True)
    groups = {}
    for group in ('train','dev'):
        records = load_records(group)
        patches = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            for i, result in enumerate(pool.map(create_patches, records), 1):
                patches.extend(result)
                if i % 25 == 0 or i == len(records):
                    print('expert_%s_images=%d/%d' % (group,i,len(records)), flush=True)
        groups[group] = patches
    tmp = PATCH_CACHE/'manifest.json.tmp'
    tmp.write_text(json.dumps({'version':1,'patch_size':PATCH,'groups':groups},indent=2),encoding='utf-8')
    os.replace(tmp, PATCH_CACHE/'manifest.json')
    print(json.dumps({'patches':{k:len(v) for k,v in groups.items()}}), flush=True)


class ExpertPatches(Dataset):
    def __init__(self, group, augment=False):
        groups = json.loads((PATCH_CACHE/'manifest.json').read_text(encoding='utf-8'))['groups']
        self.records = groups['train'] + groups['dev'] if group == 'train_all' else groups[group]
        self.augment = augment
        self.jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        image = np.load(PATCH_CACHE/'images'/(r['key']+'.npy'),allow_pickle=False)
        mask = np.load(PATCH_CACHE/'masks'/(r['key']+'.npy'),allow_pickle=False)
        image = Image.fromarray(image)
        if self.augment and random.random()<0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = np.ascontiguousarray(mask[:,::-1])
        if self.augment and random.random()<0.5:
            image = self.jitter(image)
        x = torch.from_numpy(np.array(image,dtype=np.uint8)).permute(2,0,1).float().div_(255)
        return (x-MEAN)/STD, torch.from_numpy(mask.astype(np.int64)), r['key']

if __name__ == '__main__':
    prepare()
