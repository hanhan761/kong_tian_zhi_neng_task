"""Fine-tune the human expert using split-scoped mined negatives."""
import random
import numpy as np
from PIL import Image,ImageEnhance
import torch
import v3_train
from v2_data import ROOT,TrainingImages,tensor_image
from v3_data import HardNegativeImages
class ExpertImages(HardNegativeImages):
 def __getitem__(self,index):
  if not self.hard or random.random()>=.5: return TrainingImages.__getitem__(self,index)
  record,coords=random.choice(self.hard)
  image=np.load(ROOT/'images'/(record['key']+'.npy'),mmap_mode='r',allow_pickle=False)
  mask=np.load(ROOT/'masks'/(record['key']+'.npy'),mmap_mode='r',allow_pickle=False)
  h,w=mask.shape;cy,cx=divmod(random.choice(coords),w);side=512
  y=max(0,min(cy-side//2+random.randint(-128,128),h-side));x=max(0,min(cx-side//2+random.randint(-128,128),w-side))
  image=np.array(image[y:y+side,x:x+side],copy=True);mask=np.array(mask[y:y+side,x:x+side],copy=True)
  if random.random()<.5: image=np.ascontiguousarray(image[:,::-1]);mask=np.ascontiguousarray(mask[:,::-1])
  pil=Image.fromarray(image)
  if random.random()<.5:
   pil=ImageEnhance.Brightness(pil).enhance(random.uniform(.8,1.2));pil=ImageEnhance.Color(pil).enhance(random.uniform(.8,1.2))
  mask=np.where(mask==255,255,(mask==6).astype(np.uint8))
  return tensor_image(pil),torch.from_numpy(mask.astype(np.int64))
if __name__=='__main__':
 v3_train.HardNegativeImages=ExpertImages
 v3_train.main()
