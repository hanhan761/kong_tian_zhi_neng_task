"""AeroScapes binary-person adapter; mixed with split-scoped UAVid sampling."""
import io,json,random,zipfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageEnhance
import torch
import v3_train
from v2_data import tensor_image
from v8_expert_train import SmallExpertImages
ARCHIVE=Path('/data/aerospace-intelligence/cache/aeroscapes-v8-pilot.zip')
AUDIT=Path('/data/aerospace-intelligence/cache/aeroscapes-v8-audit.json')
class ExternalExpertImages(SmallExpertImages):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs); obj=json.loads(AUDIT.read_text());self.external=[x for x in obj['accepted'] if x['person_pixels']>0]
  if len(self.external)<20:raise ValueError('too few audited external person samples')
 def __getitem__(self,index):
  if random.random()>=.25:return super().__getitem__(index)
  r=random.choice(self.external)
  with zipfile.ZipFile(ARCHIVE) as z:
   image=np.asarray(Image.open(io.BytesIO(z.read(r['image']))).convert('RGB'));mask=np.asarray(Image.open(io.BytesIO(z.read(r['mask']))))
  yx=np.argwhere(mask==1);cy,cx=yx[random.randrange(len(yx))];h,w=mask.shape;side=random.choice((320,384,512));y=max(0,min(int(cy)-side//2+random.randint(-side//4,side//4),h-side));x=max(0,min(int(cx)-side//2+random.randint(-side//4,side//4),w-side))
  im=np.array(image[y:y+side,x:x+side],copy=True);mask=np.array(mask[y:y+side,x:x+side],copy=True)
  if random.random()<.5:im=np.ascontiguousarray(im[:,::-1]);mask=np.ascontiguousarray(mask[:,::-1])
  im=Image.fromarray(im).resize((512,512),Image.Resampling.BILINEAR);mask=np.asarray(Image.fromarray(mask).resize((512,512),Image.Resampling.NEAREST))
  if random.random()<.5:im=ImageEnhance.Brightness(im).enhance(random.uniform(.8,1.2));im=ImageEnhance.Color(im).enhance(random.uniform(.8,1.2))
  target=np.where(mask==255,255,(mask==1).astype(np.uint8));return tensor_image(im),torch.from_numpy(target.astype(np.int64))
if __name__=='__main__':
 v3_train.HardNegativeImages=ExternalExpertImages
 v3_train.main()
