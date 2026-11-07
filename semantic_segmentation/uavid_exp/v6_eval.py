"""Paired nested evaluation of DINO and person expert against fourth-round fusion."""
from pathlib import Path
import argparse,json
import numpy as np
from scipy import ndimage
import torch
from train import confusion_add,score_confusion
from v2_data import atomic_json
from v2_infer import DEVICE,image_for,mask_for,load_model
from v3_data import nested_split
from v3_eval import inner_records
from v5_infer import probabilities

BASE={'global':.125,'tile':.125,'boost':.25,'dino':.5}
def variants(models):
 out={'baseline':{'type':'mix','weights':BASE,'family':'baseline'}}
 for model in ('dinob','seg'):
  if model in models:
   for w in (.25,.5):out[model+'_'+str(w)]={'type':'mix','weights':{**{k:v*(1-w) for k,v in BASE.items()},model:w},'family':model}
 out['tta_flip']={'type':'tta','weights':BASE,'family':'tta'}
 if 'expert' in models:
  for q in (.3,.4,.5,.6,.7):
   for t in (0.,.01,.05):
    for allow in ([0,2],[0,2,5]):
     name='expert_%s_%s_%s'%(q,t,len(allow))
     out[name]={'type':'expert','weights':BASE,'expert':'expert','threshold':q,'person_min':t,'allowed':allow,'family':'expert'}
 return out
@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--fold',type=int,required=True);p.add_argument('--stage',choices=('inner','outer'),required=True)
 p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--limit',type=int);a=p.parse_args()
 config=json.loads(Path(a.config).read_text());models={k:load_model(v['model'],v['checkpoint']) for k,v in config['models'].items()}
 if not set(BASE).issubset(models): raise ValueError('fifth-round baseline missing')
 split=nested_split(a.fold);records=inner_records(split['inner_val']) if a.stage=='inner' else split['outer_val']
 if a.limit is not None: records=records[:a.limit]
 cand=variants(models) if a.stage=='inner' else {'baseline':variants(models)['baseline'],**config['choices']}
 counts={name:{} for name in cand};diag={'images':0,'human_pixels':0,'missed_human':0,'human_confused':{},'per_image':[],'human_size_bins':{k:{'components':0,'pixels':0,'detected_pixels':0} for k in ('small_lt64','medium_64_255','large_ge256')}}
 needed=set(BASE)
 for item in cand.values():
  needed.update(k for k,v in item.get('weights',{}).items() if v)
  if item['type']=='expert':needed.add(item['expert'])
 for i,rec in enumerate(records,1):
  image=image_for(rec);y=torch.from_numpy(np.array(mask_for(rec),dtype=np.int64)).to(DEVICE)
  probs={k:probabilities(models[k],image,'global' if k=='global' else 'tile',expert=k=='expert') for k in sorted(needed)}
  teacher=sum(probs[k]*v for k,v in BASE.items());base_pred=teacher.argmax(0)
  diag['images']+=1;diag['human_pixels']+=int((y==6).sum());diag['missed_human']+=int(((y==6)&(base_pred!=6)).sum())
  diag['per_image'].append({'key':rec['key'],'seq':rec['seq'],'human_pixels':int((y==6).sum()),'missed_human':int(((y==6)&(base_pred!=6)).sum()),'false_human':int(((y!=6)&(y<8)&(base_pred==6)).sum())})
  human=(np.asarray(mask_for(rec))==6);labels,ncomp=ndimage.label(human)
  if ncomp:
   sizes=np.bincount(labels.ravel());det=np.bincount(labels.ravel(),weights=(base_pred.cpu().numpy()==6).ravel(),minlength=len(sizes))
   for label in range(1,len(sizes)):
    size=int(sizes[label]);key='small_lt64' if size<64 else ('medium_64_255' if size<256 else 'large_ge256')
    diag['human_size_bins'][key]['components']+=1;diag['human_size_bins'][key]['pixels']+=size;diag['human_size_bins'][key]['detected_pixels']+=int(det[label])
  for cls in range(8): diag['human_confused'][str(cls)]=diag['human_confused'].get(str(cls),0)+int(((y==6)&(base_pred==cls)).sum())
  flipped=None
  if any(cfg['type']=='tta' for cfg in cand.values()):
   trans=np.ascontiguousarray(image[:,::-1]);flipped=sum(probabilities(models[k],trans,'global' if k=='global' else 'tile').flip(-1)*v for k,v in BASE.items())
  for name,cfg in cand.items():
   if cfg['type']=='tta':pred=((teacher+flipped)/2).argmax(0)
   elif cfg['type']=='mix': pred=sum(probs[k]*v for k,v in cfg['weights'].items() if v).argmax(0)
   elif cfg['type']=='expert':
    allowed=torch.zeros_like(base_pred,dtype=torch.bool)
    for c in cfg['allowed']: allowed |= base_pred==c
    gate=allowed&(probs[cfg['expert']][1]>=cfg['threshold'])&(teacher[6]>=cfg['person_min'])
    pred=torch.where(gate,torch.full_like(base_pred,6),base_pred)
   else: raise ValueError('unknown variant')
   seq=rec['seq']; counts[name].setdefault(seq,torch.zeros((8,8),dtype=torch.int64,device=DEVICE))
   counts[name][seq]=confusion_add(counts[name][seq],y,pred)
  if i%5==0 or i==len(records): print(json.dumps({'stage':a.stage+'_eval','fold':a.fold,'images':i,'total':len(records)}),flush=True)
 result={'fold':a.fold,'stage':a.stage,'images':len(records),'diagnostic':diag,'variants':{}}
 for name,cfg in cand.items():
  m=sum(counts[name].values(),torch.zeros((8,8),dtype=torch.int64,device=DEVICE))
  result['variants'][name]={'config':cfg,'metrics':score_confusion(m),'sequence_confusions':{s:t.cpu().tolist() for s,t in counts[name].items()}}
 if a.stage=='inner':
  baseline=result['variants']['baseline']['metrics'];choices={}
  for family in ('dinob','seg','expert','tta'):
   rows=[(name,row) for name,row in result['variants'].items() if row['config']['family']==family]
   rows=[(name,row) for name,row in rows if (row['metrics']['iou']['human'] or 0)>=(baseline['iou']['human'] or 0)-.02]
   if rows:
    name,row=max(rows,key=lambda x:x[1]['metrics']['miou']);choices[family]={'name':name,'config':row['config'],
     'inner_gain':row['metrics']['miou']-baseline['miou'],'human_gain':(row['metrics']['iou']['human'] or 0)-(baseline['iou']['human'] or 0)}
  result['choices']=choices
 atomic_json(a.out,result)
if __name__=='__main__':main()
