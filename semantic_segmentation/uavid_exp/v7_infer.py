"""DINO inference with 518px overlapping native-resolution tiles."""
import torch
from v2_data import tensor_image
from v2_infer import DEVICE,AMP,positions,probabilities as old_probabilities
@torch.inference_mode()
def probabilities(model,image,mode,expert=False):
    name=model.__class__.__name__
    if name not in ('DinoSmallSeg','DinoBaseSeg','DinoLargeSeg','MaxVitSeg'): return old_probabilities(model,image,mode,expert=expert)
    h,w=image.shape[:2];size=512 if name=='MaxVitSeg' else 518;stride=384 if size==512 else 389
    axis=torch.hann_window(size,periodic=False,device=DEVICE).clamp_min_(.05)
    window=(axis[:,None]*axis[None,:]).clamp_min_(.01)
    total=torch.zeros((8,h,w),device=DEVICE); weights=torch.zeros((h,w),device=DEVICE)
    for y in positions(h,size,stride):
        for x in positions(w,size,stride):
            tile=tensor_image(image[y:y+size,x:x+size]).unsqueeze(0).to(DEVICE)
            with torch.cuda.amp.autocast(dtype=AMP): logits=model(tile)
            total[:,y:y+size,x:x+size]+=logits.float().softmax(1)[0]*window
            weights[y:y+size,x:x+size]+=window
    if float(weights.min())<=0: raise RuntimeError('uncovered image pixels')
    return total/weights.unsqueeze(0)
