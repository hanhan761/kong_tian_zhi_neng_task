"""Paired evaluation against the seventh-round DINO-L fusion."""
from pathlib import Path
import argparse,json
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from models import forward
from v2_data import atomic_json,tensor_image
from v2_infer import DEVICE,AMP,positions,image_for,mask_for,load_model
from v3_data import nested_split
from v3_eval import inner_records
from v7_infer import probabilities
from train import confusion_add,score_confusion
R=Path('/data/aerospace-intelligence/runs/uavid-20261106-v7-error-attack')
@torch.inference_mode()
def small_probs(model,image):
 h,w=image.shape[:2];size=384;stride=288
 a=torch.hann_window(size,periodic=False,device=DEVICE).clamp_min_(.05);window=(a[:,None]*a[None,:]).clamp_min_(.01)
 total=torch.zeros((h,w),device=DEVICE);weights=torch.zeros((h,w),device=DEVICE)
 for y in positions(h,size,stride):
  for x in positions(w,size,stride):
   im=Image.fromarray(np.asarray(image[y:y+size,x:x+size])).resize((512,512),Image.Resampling.BILINEAR)
   z=tensor_image(im).unsqueeze(0).to(DEVICE)
   with torch.cuda.amp.autocast(dtype=AMP): logits,_=forward(model,z)
   p=F.interpolate(logits.float().softmax(1)[:,1:2],size=(size,size),mode='bilinear',align_corners=False)[0,0]
   total[y:y+size,x:x+size]+=p*window;weights[y:y+size,x:x+size]+=window
 if float(weights.min())<=0:raise ValueError('uncovered expert tile')
 return total/weights
@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--fold',type=int,required=True);p.add_argument('--stage',choices=['inner','outer'],required=True);p.add_argument('--expert');p.add_argument('--out',required=True);p.add_argument('--limit',type=int);a=p.parse_args()
 c=json.loads((R/('eval_%s_%d_config.json'%(a.stage,a.fold))).read_text()); inner=json.loads((R/('eval_inner_%d.json'%a.fold)).read_text());weights=inner['choices']['dinol']['config']['weights']
 models={k:load_model(c['models'][k]['model'],c['models'][k]['checkpoint']) for k,v in weights.items() if v}
 expert=load_model('expert_person',a.expert) if a.expert else None
 records=inner_records(nested_split(a.fold)['inner_val']) if a.stage=='inner' else nested_split(a.fold)['outer_val']
 if a.limit:records=records[:a.limit]
 names=['baseline','flip']+(['expert_%.1f'%q for q in (.4,.6,.8)]+['flip_expert_%.1f'%q for q in (.4,.6,.8)] if expert else [])
 counts={n:{} for n in names}
 for i,r in enumerate(records,1):
  image=image_for(r); truth=torch.from_numpy(np.array(mask_for(r),dtype=np.int64)).to(DEVICE)
  def mix(im):return sum(probabilities(models[k],im,'global' if k=='global' else 'tile')*v for k,v in weights.items() if v)
  base=mix(image);flipped=mix(np.ascontiguousarray(image[:,::-1])).flip(-1);tta=(base+flipped)/2
  ep=small_probs(expert,image) if expert else None
  preds={'baseline':base.argmax(0),'flip':tta.argmax(0)}
  if expert:
   for typ,teacher in [('expert',base),('flip_expert',tta)]:
    pred=teacher.argmax(0);allowed=(pred==0)|(pred==2)|(pred==5)
    for threshold in (.4,.6,.8):
     gate=allowed&(ep>=threshold)&(teacher[6]>=.01)
     preds['%s_%.1f'%(typ,threshold)]=torch.where(gate,torch.full_like(pred,6),pred)
  for name,pred in preds.items():
   counts[name].setdefault(r['seq'],torch.zeros((8,8),dtype=torch.int64,device=DEVICE));counts[name][r['seq']]=confusion_add(counts[name][r['seq']],truth,pred)
  if i%5==0 or i==len(records):print(json.dumps({'stage':a.stage,'fold':a.fold,'images':i,'total':len(records)}),flush=True)
 result={'fold':a.fold,'stage':a.stage,'images':len(records),'weights':weights,'variants':{}}
 for n in names:
  m=sum(counts[n].values(),torch.zeros((8,8),dtype=torch.int64,device=DEVICE)); result['variants'][n]={'metrics':score_confusion(m),'sequence_confusions':{s:t.cpu().tolist() for s,t in counts[n].items()}}
 if a.limit is None:
  old=json.loads((R/('eval_%s_%d.json'%(a.stage,a.fold))).read_text());name=inner['choices']['dinol']['name'] if a.stage=='inner' else 'dinol'
  if result['variants']['baseline']['sequence_confusions']!=old['variants'][name]['sequence_confusions']:raise ValueError('seventh-round baseline replay mismatch')
 atomic_json(a.out,result)
if __name__=='__main__':main()
