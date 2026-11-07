"""Score frozen predictions against a pixel-verified public UAVid mirror."""
from pathlib import Path
import argparse, hashlib, json, tarfile
from io import BytesIO
import numpy as np
from PIL import Image
import torch
from train import score_confusion
from v2_data import INDEX, atomic_json

REV='f6b176dfc880d905c62957dcba820820e725520e'
def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def open_archive(path, expected):
    if file_hash(path)!=expected: raise ValueError('archive transfer checksum mismatch')
    tar=tarfile.open(path,'r')
    for m in tar.getmembers():
        if not m.isfile() or '/' in m.name or not m.name.endswith('.png') or m.size>15000000:
            raise ValueError('unexpected archive member')
    if len({m.name for m in tar.getmembers()})!=len(tar.getmembers()): raise ValueError('duplicate archive member')
    return tar

def mask(tar, name):
    stream=tar.extractfile(name)
    if stream is None: raise ValueError('mask missing')
    image=np.asarray(Image.open(BytesIO(stream.read())))
    if image.ndim==3:
        image=(image[:,:,:3].astype(np.uint32)*np.array([65536,256,1],dtype=np.uint32)).sum(2)
    if image.ndim!=2: raise ValueError('unexpected mask format')
    return image

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--freeze',required=True);p.add_argument('--train-tar',required=True);p.add_argument('--train-sha',required=True)
    p.add_argument('--test-tar');p.add_argument('--test-sha');p.add_argument('--v2');p.add_argument('--v3');p.add_argument('--out')
    p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    freeze=Path(a.freeze); freeze_obj=json.loads(freeze.read_text(encoding='utf-8'))
    freezehash=file_hash(freeze)
    expected=json.loads((freeze.parent/'freeze_sha256.json').read_text())['sha256']
    if freezehash!=expected or not freeze_obj.get('frozen_before_official_and_test'): raise ValueError('frozen selection checksum mismatch')
    for path,digest in freeze_obj['checkpoint_sha256'].items():
        if file_hash(path)!=digest: raise ValueError('frozen model checkpoint changed')
    index=json.loads(INDEX.read_text(encoding='utf-8'))['items']
    chosen=[]
    for seq in sorted({x['seq'] for x in index})[:10]:
        chosen.append(max((x for x in index if x['seq']==seq),key=lambda x:x['counts']['6']))
    mapping={}
    with open_archive(a.train_tar,a.train_sha) as tar:
        expected_names={r['key'].removeprefix('train_')+'.png' for r in chosen}
        if {m.name for m in tar.getmembers()}!=expected_names: raise ValueError('unexpected train verification set')
        for i,rec in enumerate(chosen,1):
            mirror=mask(tar,rec['key'].removeprefix('train_')+'.png')
            local=np.load('/data/aerospace-intelligence/cache/uavid-native-v2/masks/'+rec['key']+'.npy',mmap_mode='r')
            if mirror.shape!=local.shape: raise ValueError('training mask shape mismatch')
            for value in np.unique(mirror):
                vals=np.unique(local[mirror==value])
                if len(vals)!=1: raise ValueError('mirror does not match original training mask')
                key,val=int(value),int(vals[0])
                if key in mapping and mapping[key]!=val: raise ValueError('inconsistent mirror class mapping')
                mapping[key]=val
            print(json.dumps({'stage':'training_mask_verification','images':i,'total':len(chosen)}),flush=True)
    if not set(range(8)).issubset(set(mapping.values())): raise ValueError('training verification lacks classes')
    # 255 is the declared ignore ID for this Ultralytics mask format.
    # The project also excludes 255 from segmentation metrics.
    mapping[255]=255
    print(json.dumps({'stage':'verification_passed','images':len(chosen),'mapping':mapping,'freeze_sha256':freezehash}),flush=True)
    if a.verify_only: return
    if not all((a.test_tar,a.test_sha,a.v2,a.v3,a.out)): raise ValueError('test arguments missing')
    predpaths=sorted(Path(a.v2).glob('seq*/*.png'))
    if len(predpaths)!=150: raise ValueError('unexpected test prediction count')
    names={x.parent.name+'_'+x.stem+'.png' for x in predpaths}
    mats={'v2':np.zeros((8,8),np.int64),'v3':np.zeros((8,8),np.int64)}
    ignored_pixels=0
    with open_archive(a.test_tar,a.test_sha) as tar:
        if {m.name for m in tar.getmembers()}!=names: raise ValueError('test mask names/count mismatch')
        for i,file in enumerate(predpaths,1):
            truth0=mask(tar,file.parent.name+'_'+file.stem+'.png')
            truth=np.full(truth0.shape,255,np.uint8)
            for value in np.unique(truth0):
                if int(value) not in mapping: raise ValueError('unverified test mask value')
                truth[truth0==value]=mapping[int(value)]
            ignored_pixels += int((truth == 255).sum())
            for name,base in [('v2',a.v2),('v3',a.v3)]:
                pred=np.asarray(Image.open(Path(base)/file.parent.name/file.name))
                if pred.shape!=truth.shape or pred.max()>7: raise ValueError('prediction shape or class mismatch')
                valid=truth<8
                mats[name]+=np.bincount((truth[valid].astype(np.int64)*8+pred[valid]).ravel(),minlength=64).reshape(8,8)
            if i%10==0: print(json.dumps({'stage':'public_test','images':i,'total':150}),flush=True)
    atomic_json(a.out,{'images':150,'selection_sha256_before_test':freezehash,'source_revision':REV,
        'source':'https://huggingface.co/datasets/dronefreak/UAVid-2020',
        'source_status':'unofficial mirror, 10 training masks pixel-verified', 'mapping':mapping,
        'test_archive_sha256':a.test_sha,'ignored_pixels':ignored_pixels,'models':{k:score_confusion(torch.from_numpy(v)) for k,v in mats.items()}})
if __name__=='__main__': main()
