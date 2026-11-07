"""Native 518-pixel DINO crops, with split-scoped rare-object sampling."""
import random
import numpy as np
from PIL import Image, ImageEnhance
import torch
from v2_data import ROOT, RARE, tensor_image
from v3_data import HardNegativeImages
class MaxVitImages(HardNegativeImages):
    def __getitem__(self,index):
        if self.expert or self.mode!='tile': raise ValueError('MaxViT requires multiclass tiles')
        record=random.choice(self.records); cls=None
        if random.random()<self.rare_fraction:
            cls=random.choices(RARE,weights=(2,1,1),k=1)[0]
            if self.by_class[cls]: record=random.choice(self.by_class[cls])
            else: cls=None
        im=np.load(ROOT/'images'/(record['key']+'.npy'),mmap_mode='r',allow_pickle=False)
        mask=np.load(ROOT/'masks'/(record['key']+'.npy'),mmap_mode='r',allow_pickle=False)
        h,w=mask.shape; side=512
        if cls is not None:
            cy,cx=divmod(random.choice(record['coords'][str(cls)]),w)
            y=max(0,min(cy-side//2+random.randint(-side//4,side//4),h-side))
            x=max(0,min(cx-side//2+random.randint(-side//4,side//4),w-side))
        else:
            y=random.randint(0,max(0,h-side));x=random.randint(0,max(0,w-side))
        image=np.array(im[y:y+side,x:x+side],copy=True); labels=np.array(mask[y:y+side,x:x+side],copy=True)
        if random.random()<.5: image=np.ascontiguousarray(image[:,::-1]);labels=np.ascontiguousarray(labels[:,::-1])
        pil=Image.fromarray(image)
        if random.random()<.5:
            pil=ImageEnhance.Brightness(pil).enhance(random.uniform(.8,1.2))
            pil=ImageEnhance.Color(pil).enhance(random.uniform(.8,1.2))
        return tensor_image(pil),torch.from_numpy(labels.astype(np.int64))
