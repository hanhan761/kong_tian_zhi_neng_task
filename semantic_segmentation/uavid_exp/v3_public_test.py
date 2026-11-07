"""One-shot scoring from a pixel-verified public mirror, only after freeze."""
from pathlib import Path
import argparse, hashlib, json, time, urllib.request
import numpy as np
from PIL import Image
from io import BytesIO
from data import DATA
from train import score_confusion
from v2_data import INDEX, atomic_json
import torch

MIRROR = 'https://huggingface.co/datasets/dronefreak/UAVid-2020/resolve/main/masks/'
def get(path):
    req = urllib.request.Request(MIRROR+path, headers={'User-Agent':'UAVid-research-evaluation/1.0'})
    last = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=25) as response:
                data = response.read(15000000)
            break
        except (OSError, urllib.error.URLError) as exc:
            last = exc
            if attempt == 2: raise
            time.sleep(2 ** attempt)
    image = np.asarray(Image.open(BytesIO(data)))
    if image.ndim == 3:
        image = (image[:,:,:3].astype(np.uint32) * np.array([65536,256,1],dtype=np.uint32)).sum(2)
    if image.ndim != 2: raise ValueError('unexpected mask format')
    return image

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--freeze',required=True)
    p.add_argument('--v2',required=True)
    p.add_argument('--v3',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    freeze=Path(a.freeze)
    if not freeze.is_file(): raise ValueError('selection has not been frozen')
    frozen_hash=hashlib.sha256(freeze.read_bytes()).hexdigest()
    index=json.loads(INDEX.read_text(encoding='utf-8'))['items']
    # Diverse known train sequences establish pixel mapping, including all classes.
    chosen=[]
    for seq in sorted({x['seq'] for x in index}):
        group=[x for x in index if x['seq']==seq]
        chosen.append(max(group,key=lambda x:x['counts']['6']))
        if len(chosen)==10: break
    mapping={}
    for rec in chosen:
        mirror=get('train/'+rec['key'].removeprefix('train_')+'.png')
        local=np.load('/data/aerospace-intelligence/cache/uavid-native-v2/masks/'+rec['key']+'.npy',mmap_mode='r')
        if mirror.shape != local.shape: raise ValueError('mirror training shape mismatch')
        for value in np.unique(mirror):
            values=np.unique(local[mirror==value])
            if len(values)!=1: raise ValueError('mirror does not match original train masks')
            val=int(values[0]); key=int(value)
            if key in mapping and mapping[key]!=val: raise ValueError('inconsistent mirror class mapping')
            mapping[key]=val
    if not set(range(8)).issubset(set(mapping.values())): raise ValueError('mirror verification lacks some classes')
    # UAVid mirror's Ultralytics format uses 255 for ignored pixels.
    mapping[255]=255
    files=sorted((DATA/'test').glob('seq*/Images/*.png'))
    if len(files)!=150: raise ValueError('test count mismatch')
    mats={'v2':np.zeros((8,8),np.int64),'v3':np.zeros((8,8),np.int64)}
    for i,file in enumerate(files,1):
        key=file.parent.parent.name+'_'+file.stem
        remote=get('test/'+key+'.png')
        truth=np.full(remote.shape,255,np.uint8)
        for value in np.unique(remote):
            if int(value) not in mapping: raise ValueError('unverified test mask value')
            truth[remote==value]=mapping[int(value)]
        for name,base in [('v2',a.v2),('v3',a.v3)]:
            pred=np.asarray(Image.open(Path(base)/file.parent.parent.name/file.name))
            if pred.shape!=truth.shape or pred.max()>7: raise ValueError('prediction shape or class mismatch')
            valid=truth<8
            mats[name]+=np.bincount((truth[valid].astype(np.int64)*8+pred[valid]).ravel(),minlength=64).reshape(8,8)
        if i%10==0: print(json.dumps({'stage':'public_test','images':i}),flush=True)
    atomic_json(a.out, {'images':150,'selection_sha256_before_test':frozen_hash,
        'source':MIRROR, 'source_status':'unofficial mirror, 10 train masks pixel-verified against local originals',
        'mapping':mapping, 'models':{k:score_confusion(torch.from_numpy(v)) for k,v in mats.items()}})
if __name__=='__main__': main()
