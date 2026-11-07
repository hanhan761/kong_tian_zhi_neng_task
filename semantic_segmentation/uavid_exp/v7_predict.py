"""Predict reference images with a frozen fifth-round strategy."""
from pathlib import Path
import argparse,json,time
import numpy as np
from PIL import Image
import torch
from data import DATA,PALETTE,load_records
from train import confusion_add,score_confusion
from v2_data import atomic_json
from v2_infer import DEVICE,image_for,mask_for,load_model
from v7_infer import probabilities
@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--selection',required=True);p.add_argument('--out',required=True);p.add_argument('--deadline',type=float,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 sel=json.loads(Path(a.selection).read_text());models={k:load_model(v['model'],v['checkpoint']) for k,v in sel['models'].items()};cfg=sel['strategy'];out=Path(a.out)
 def predict(image):
  probs={k:probabilities(models[k],image,'global' if k=='global' else 'tile') for k,v in cfg['weights'].items() if v}
  teacher=sum(probs[k]*v for k,v in cfg['weights'].items() if v)
  if cfg['type']=='tta':
   transformed=np.ascontiguousarray(image[:,::-1])
   flip=sum(probabilities(models[k],transformed,'global' if k=='global' else 'tile').flip(-1)*v for k,v in cfg['weights'].items() if v)
   teacher=(teacher+flip)/2
  if cfg['type']=='calibrate':teacher[6]*=cfg['human_factor']
  pred=teacher.argmax(0)
  if cfg['type']=='expert':
   q=probabilities(models[cfg['expert']],image,'tile',expert=True)[1]
   allowed=torch.zeros_like(pred,dtype=torch.bool)
   for c in cfg['allowed']: allowed|=pred==c
   pred=torch.where(allowed&(q>=cfg['threshold'])&(teacher[6]>=cfg['person_min']),torch.full_like(pred,6),pred)
  return pred
 val=load_records('official_val')
 if len(val)!=70: raise ValueError('wrong validation count')
 if a.smoke:
  pred=predict(image_for(val[0]));assert tuple(pred.shape)==tuple(image_for(val[0]).shape[:2]);atomic_json(out/'smoke.json',{'shape':list(pred.shape)});return
 mat=torch.zeros((8,8),dtype=torch.int64,device=DEVICE)
 for i,r in enumerate(val,1):
  if time.time()>=a.deadline: raise TimeoutError('deadline')
  pred=predict(image_for(r));truth=torch.from_numpy(np.array(mask_for(r),dtype=np.int64)).to(DEVICE);mat=confusion_add(mat,truth,pred)
  if i%10==0:print(json.dumps({'stage':'validation','images':i}),flush=True)
 atomic_json(out/'official_validation.json',{'images':70,'metrics':score_confusion(mat),'qualification':'previously observed, development reference'})
 files=sorted((DATA/'test').glob('seq*/Images/*.png'))
 if len(files)!=150:raise ValueError('wrong test count')
 pal=np.asarray(PALETTE,dtype=np.uint8)
 for i,f in enumerate(files,1):
  if time.time()>=a.deadline:raise TimeoutError('deadline')
  with Image.open(f) as im: image=np.asarray(im.convert('RGB'))
  pred=predict(image).cpu().numpy().astype(np.uint8)
  for kind,arr,mode in (('trainids',pred,'L'),('color',pal[pred],'RGB')):
   dest=out/'test_predictions'/kind/f.parent.parent.name/f.name;dest.parent.mkdir(parents=True,exist_ok=True)
   tmp=dest.with_suffix('.png.tmp');Image.fromarray(arr,mode=mode).save(tmp,format='PNG');tmp.replace(dest)
  if i%10==0:print(json.dumps({'stage':'test_export','images':i}),flush=True)
 atomic_json(out/'prediction_summary.json',{'test_images':150,'path':str(out/'test_predictions')})
if __name__=='__main__':main()
