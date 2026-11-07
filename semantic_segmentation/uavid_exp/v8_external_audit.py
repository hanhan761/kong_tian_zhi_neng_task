"""Validate the fixed AeroScapes sample and screen obvious UAVid overlap."""
import hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from v2_data import atomic_json
P=Path('/data/aerospace-intelligence');D=P/'data/semantic-segmentation/course-source/datasets/data';Z=P/'cache/aeroscapes-v8-pilot.zip';OUT=P/'cache/aeroscapes-v8-audit.json'
def digest(raw):return hashlib.sha256(raw).hexdigest()
def phash(im):
 a=np.asarray(im.convert('L').resize((17,16),Image.Resampling.BILINEAR));return int.from_bytes(np.packbits((a[:,1:]>a[:,:-1]).ravel()).tobytes(),'big')
raw=Z.read_bytes();assert digest(raw)=='5fb59d9f01fe657d1d845a9a65b1fc8ead7203289a790262a9746b7820e1a23c'
files=[]
for part in ('train','valid','test'):files.extend(sorted((D/part).glob('seq*/Images/*.png')))
if len(files)!=420:raise ValueError('unexpected UAVid reference image count')
refs=[]
for i,p in enumerate(files,1):
 with Image.open(p) as im:refs.append(phash(im))
 if i%100==0:print('overlap_refs',i,flush=True)
accepted=[];rejected=[]
with zipfile.ZipFile(io.BytesIO(raw)) as z:
 m=json.loads(z.read('manifest.json'));assert m['revision']=='7c2e6f4e301882934799753989563d85ecd6157f' and len(m['items'])==160
 for item in m['items']:
  a=z.read(item['image']);b=z.read(item['mask']);assert digest(a)==item['image_sha256'] and digest(b)==item['mask_sha256'];h=phash(Image.open(io.BytesIO(a)))
  dist=min(bin(h^x).count('1') for x in refs)
  (rejected if dist<=4 else accepted).append(item)
if sum(x['person_pixels']>0 for x in accepted)<20:raise ValueError('too few clean person images')
atomic_json(OUT,{'source':m['source'],'revision':m['revision'],'license':m['license'],'archive_sha256':digest(raw),'screen':'256-bit dHash distance <=4 against 420 UAVid images; cannot rule out all scene-level overlap','reference_images':len(files),'accepted':accepted,'rejected':len(rejected)})
print(json.dumps({'stage':'external_audit_complete','accepted':len(accepted),'rejected':len(rejected),'person_images':sum(x['person_pixels']>0 for x in accepted)}),flush=True)
