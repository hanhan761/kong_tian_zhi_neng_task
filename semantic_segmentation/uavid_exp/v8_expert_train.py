"""High resolution, split-scoped human expert with mined background negatives."""
import random
import numpy as np
from PIL import Image, ImageEnhance
import torch
import v3_train
from v2_data import ROOT, tensor_image
from v3_data import HardNegativeImages

class SmallExpertImages(HardNegativeImages):
 def __getitem__(self,index):
  if self.hard and random.random()<.35:
   record,coords=random.choice(self.hard); focus=random.choice(coords)
  elif random.random()<.8:
   record=random.choice(self.by_class[6]); focus=random.choice(record['coords']['6'])
  else:
   record=random.choice(self.records); focus=None
  img=np.load(ROOT/'images'/(record['key']+'.npy'),mmap_mode='r',allow_pickle=False)
  mask=np.load(ROOT/'masks'/(record['key']+'.npy'),mmap_mode='r',allow_pickle=False)
  h,w=mask.shape; side=random.choice((320,384,512))
  if focus is None: y=random.randint(0,h-side);x=random.randint(0,w-side)
  else:
   cy,cx=divmod(focus,w); y=max(0,min(cy-side//2+random.randint(-side//4,side//4),h-side)); x=max(0,min(cx-side//2+random.randint(-side//4,side//4),w-side))
  img=np.array(img[y:y+side,x:x+side],copy=True);mask=np.array(mask[y:y+side,x:x+side],copy=True)
  if random.random()<.5: img=np.ascontiguousarray(img[:,::-1]);mask=np.ascontiguousarray(mask[:,::-1])
  im=Image.fromarray(img).resize((512,512),Image.Resampling.BILINEAR)
  mask=np.asarray(Image.fromarray(mask).resize((512,512),Image.Resampling.NEAREST))
  if random.random()<.5:im=ImageEnhance.Brightness(im).enhance(random.uniform(.8,1.2));im=ImageEnhance.Color(im).enhance(random.uniform(.8,1.2))
  mask=np.where(mask==255,255,(mask==6).astype(np.uint8))
  return tensor_image(im),torch.from_numpy(mask.astype(np.int64))
if __name__=='__main__':
 v3_train.HardNegativeImages=SmallExpertImages
 v3_train.main()
