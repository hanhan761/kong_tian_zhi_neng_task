"""24-hour bounded, nested, paired UAVid human-recall improvement experiment."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from collections import Counter
from pathlib import Path
import hashlib,json,os,shutil,threading,time
import numpy as np
import torch
import v2_pipeline as core
from v2_data import atomic_json
ROOT=Path('/data/aerospace-intelligence');V3=ROOT/'runs/uavid-20261105-v3-error-attack';V4=ROOT/'runs/uavid-20261105-v4-human-recall';V5=ROOT/'runs/uavid-20261106-v5-breakthrough';V6=ROOT/'runs/uavid-20261106-v6-pretrained';RUN=ROOT/'runs/uavid-20261106-v7-error-attack';CODE=Path(__file__).resolve().parent
START=time.monotonic();BUDGET=23*3600+45*60; DEADLINE=time.time()+BUDGET
core.RUN,core.START,core.BUDGET,core.ABS_DEADLINE=RUN,START,BUDGET,DEADLINE
BASE={'global':.0625,'tile':.0625,'boost':.125,'dino':.25,'dinob':.5}

def stage(name,**kw): core.stage(name,**kw)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def parent(fold,phase,kind):
 if kind=='dinob': path=V6/'models'/('%s_fold%d_dinob'%(phase,fold))/'best.pt';expected='dinov2_base_seg'
 elif kind=='dino': path=V5/'models'/('%s_fold%d_dino'%(phase,fold))/'best.pt';expected='dinov2_small_seg'
 else:
  path=(V4 if kind=='boost' else V3)/'models'/('%s_fold%d_%s'%(phase,fold,'sfa' if kind=='boost' else kind))/'best.pt';expected='course_resnet50'
 obj=torch.load(path,map_location='cpu',weights_only=True);contract=json.loads((path.parent/'v3_contract.json').read_text())
 if contract['stage']!=phase or contract['fold']!=fold or obj['model_name']!=expected:raise ValueError('baseline parent mismatch')
 return {'model':expected,'checkpoint':str(path),'sha256':sha(path)}
def expert_parent(fold,phase):
 path=V6/'models'/('%s_fold%d_expert'%(phase,fold))/'best.pt'
 obj=torch.load(path,map_location='cpu',weights_only=True);contract=json.loads((path.parent/'v3_contract.json').read_text())
 if contract['stage']!=phase or contract['fold']!=fold or obj['model_name']!='expert_person':raise ValueError('expert parent mismatch')
 return path

def mine(fold,phase,checkpoint,note,end):
 out=RUN/('hard_%s_%d.json'%(phase,fold))
 if not out.exists():core.command('mine_%s_%d'%(phase,fold),[str(CODE/'v3_mine.py'),'--fold',str(fold),'--stage',phase,'--checkpoint',str(checkpoint),'--out',str(out)],note,end)
 return out

def train(fold,phase,kind,note,end,epochs,init=None,hard=None):
 folder=RUN/'models'/('%s_fold%d_%s'%(phase,fold,kind))
 if kind=='dinol':script='v7_train.py';model='dinov2_large_seg';schedule='12';samples='300';lr='.00005'
 elif kind=='maxvit':script='v7_train.py';model='unet_maxvit_small';schedule='12';samples='300';lr='.0001'
 elif kind=='expert':script='v5_expert_train.py';model='expert_person';schedule='8';samples='400';lr='.0001'
 else:raise ValueError('unknown candidate')
 args=[str(CODE/script),'--v3-stage',phase,'--model',model,'--mode','tile','--fold',str(fold),'--out',str(folder),
 '--epochs',str(epochs),'--schedule-epochs',schedule,'--samples',samples,'--batch','2','--workers','2','--lr',lr,
 '--rare-fraction','.75','--loss','ce_dice','--seed','42','--max-seconds',str(max(120,int(end-time.monotonic()-30)))]
 if init:args+=['--v3-init',str(init)]
 if hard:args+=['--v3-hard',str(hard)]
 if (folder/'last.pt').exists():args.append('--resume')
 label='%s_fold%d_%s'%(phase,fold,kind)
 for attempt in range(2):
  try:core.command(label,args,note,end);break
  except Exception:
   if attempt or time.monotonic()>end-180:raise
   if (folder/'last.pt').exists() and '--resume' not in args:args.append('--resume')
 summary=json.loads((folder/'summary.json').read_text());path=folder/'best.pt'
 if summary['epochs_completed']!=epochs:raise TimeoutError('incomplete training')
 obj=torch.load(path,map_location='cpu',weights_only=True)
 return {'model':obj['model_name'],'checkpoint':str(path),'best_epoch':int(obj['epoch'])}

def many(jobs,note,end,workers):
 out={}
 with ThreadPoolExecutor(max_workers=workers) as pool:
  fut={pool.submit(train,f,ph,k,note,end,ep,init,hard):(f,ph,k) for f,ph,k,ep,init,hard in jobs}
  for future in as_completed(fut):
   key=fut[future]
   try:out[key]=future.result()
   except Exception as e:core.record_failure(str(key),e)
 return out

def evaluate(fold,phase,models,choices,note,end):
 cfg=RUN/('eval_%s_%d_config.json'%(phase,fold));out=RUN/('eval_%s_%d.json'%(phase,fold))
 atomic_json(cfg,{'models':models,'choices':choices})
 core.command('eval_%s_%d'%(phase,fold),[str(CODE/'v7_eval.py'),'--fold',str(fold),'--stage',phase,'--config',str(cfg),'--out',str(out)],note,end)
 return json.loads(out.read_text())

def summarize(folds):
 names=set.intersection(*(set(x['variants']) for x in folds));table={}
 for name in sorted(names):
  seq={}
  for f in folds:
   for s,m in f['variants'][name]['sequence_confusions'].items():
    if s in seq:raise ValueError('outer overlap')
    seq[s]=np.asarray(m,dtype=np.int64)
  if len(seq)!=20:raise ValueError('outer sequence coverage incomplete')
  m=sum(seq.values());iv=core.ious(m)
  table[name]={'miou':float(np.nanmean(iv)),'human_iou':float(iv[6]),'class_iou':[float(x) if np.isfinite(x) else None for x in iv],
   'confusion':m.tolist(),'sequence_confusions':{k:v.tolist() for k,v in seq.items()}}
 base=table['baseline']
 for name,row in table.items():
  row['delta_miou']=row['miou']-base['miou'];row['delta_human']=row['human_iou']-base['human_iou']
  row['eligible']=name!='baseline' and row['delta_miou']>=.005 and row['delta_human']>=-.02
  keys=sorted(row['sequence_confusions']);rng=np.random.default_rng(42);diffs=[]
  for draw in rng.integers(0,len(keys),size=(1000,len(keys))):
   a=sum(np.asarray(row['sequence_confusions'][keys[i]],dtype=np.int64) for i in draw)
   b=sum(np.asarray(base['sequence_confusions'][keys[i]],dtype=np.int64) for i in draw)
   diffs.append(float(np.nanmean(core.ious(a))-np.nanmean(core.ious(b))))
  row['sequence_bootstrap_delta_miou_95']=[float(x) for x in np.quantile(diffs,[.025,.975])]
 winners=[x for x in table if table[x]['eligible']]
 return {'variants':table,'selected':max(winners,key=lambda x:table[x]['miou']) if winners else 'v6_fallback',
 'qualification':'Repeated development over sequence folds; not an independent blind evaluation.'}

def execute(note):
 pilot_end=START+3*3600;dev_end=START+15*3600;final_end=START+20*3600;eval_end=START+BUDGET-10
 parents={}
 for f in range(3):
  for phase in ('inner','outer'):
   for k in BASE:parents['%d_%s_%s'%(f,phase,k)]=parent(f,phase,k)
   e=expert_parent(f,phase);parents['%d_%s_expert'%(f,phase)]={'model':'expert_person','checkpoint':str(e),'sha256':sha(e)}
 atomic_json(RUN/'parents.json',parents);stage('pilot_training')
 ep=expert_parent(0,'inner');hard=mine(0,'inner',ep,note,pilot_end)
 pair=[(0,'inner','maxvit',1,None,None),(0,'inner','dinol',1,None,None)]
 t=time.monotonic();par=many(pair,note,pilot_end,2);parallel_s=time.monotonic()-t
 t=time.monotonic();serial=many([(f,ph,k,2,init,hn) for f,ph,k,n,init,hn in pair if (f,ph,k) in par],note,pilot_end,1);serial_s=time.monotonic()-t
 samples=[]
 if (RUN/'resources.jsonl').exists():samples=[json.loads(x)['gpu_memory_mib'] for x in (RUN/'resources.jsonl').read_text().splitlines()]
 peak=max(samples or [0]);workers=2 if len(par)==len(serial)==2 and parallel_s*1.10<serial_s and peak<20480 else 1
 atomic_json(RUN/'efficiency.json',{'parallel_first_epochs_seconds':parallel_s,'serial_second_epochs_seconds':serial_s,'peak_gpu_mib':peak,'concurrency':workers,'qualification':'different epochs; exploratory scheduling evidence'})
 jobs=[(f,ph,k,4,init,hn) for f,ph,k,n,init,hn in pair if (f,ph,k) in serial]+[(0,'inner','expert',4,ep,hard)]
 pilot=many(jobs,note,pilot_end,workers)
 if not pilot:raise RuntimeError('no pilot model completed')
 models={k:parents['0_inner_'+k] for k in BASE};models.update({k:pilot[(0,'inner',k)] for k in ('maxvit','dinol','expert') if (0,'inner',k) in pilot})
 first=evaluate(0,'inner',models,{},note,pilot_end);baseline=first['variants']['baseline']['metrics'];keep=[]
 for k in ('maxvit','dinol','expert','tta','calibrate'):
  if k not in first.get('choices',{}):continue
  c=first['choices'][k]
  if c['inner_gain']>=-.01 or c['human_gain']>=.02:keep.append(k)
 atomic_json(RUN/'pilot_decision.json',{'kept':keep,'inner_choices':first.get('choices',{}),'rule':'discard if mIoU falls >1pp and human improves <2pp','baseline':baseline})
 stage('pilot_complete',kept=keep)
 inner=dict(pilot);jobs=[]
 for f in range(3):
  for k in keep:
   if k in ('tta','calibrate'):continue
   ep=expert_parent(f,'inner') if k=='expert' else None
   hard=mine(f,'inner',ep,note,dev_end) if k=='expert' else None
   jobs.append((f,'inner',k,8 if k=='expert' else 12,ep,hard))
 more=many(jobs,note,dev_end,workers);inner.update(more)
 keep=[k for k in keep if k in ('tta','calibrate') or all((f,'inner',k) in more for f in range(3))]
 inners=[]
 for f in range(3):
  models={k:parents['%d_inner_%s'%(f,k)] for k in BASE};models.update({k:inner[(f,'inner',k)] for k in keep if k not in ('tta','calibrate')})
  inners.append(evaluate(f,'inner',models,{},note,dev_end))
 keep=[k for k in keep if all(k in row['choices'] for row in inners)];stage('outer_training',kept=keep);jobs=[]
 for f in range(3):
  for k in keep:
   if k in ('tta','calibrate'):continue
   ep=expert_parent(f,'outer') if k=='expert' else None
   hard=mine(f,'outer',ep,note,dev_end) if k=='expert' else None
   jobs.append((f,'outer',k,inner[(f,'inner',k)]['best_epoch'],ep,hard))
 outer=many(jobs,note,dev_end,workers);keep=[k for k in keep if k in ('tta','calibrate') or all((f,'outer',k) in outer for f in range(3))]
 folds=[]
 for f in range(3):
  models={k:parents['%d_outer_%s'%(f,k)] for k in BASE};models.update({k:outer[(f,'outer',k)] for k in keep if k not in ('tta','calibrate')})
  choices={k:inners[f]['choices'][k]['config'] for k in keep};folds.append(evaluate(f,'outer',models,choices,note,dev_end))
 comparison=summarize(folds);atomic_json(RUN/'cross_validation.json',comparison)
 atomic_json(RUN/'error_analysis.json',{'inner':[x['diagnostic'] for x in inners],'outer':[x['diagnostic'] for x in folds],'provenance':'baseline held out each evaluated sequence'})
 selected=comparison['selected'];stage('development_complete',selected=selected)
 if selected=='v6_fallback':
  frozen={'selected':selected,'fallback':str(V6/'selection.json'),'development':comparison,'test_labels_not_used_for_selection':True}
  atomic_json(RUN/'selection.json',frozen);atomic_json(RUN/'report.json',{'status':'partial' if core.FAILURES else 'complete','selection':frozen,'failures':core.FAILURES,'reason':'no candidate met paired threshold'})
 else:
  candidates=[r['choices'][selected] for r in inners];encoded=[json.dumps(x['config'],sort_keys=True) for x in candidates];counts=Counter(encoded);maxn=max(counts.values())
  chosen=max((s for s in counts if counts[s]==maxn),key=lambda s:sum(x['inner_gain'] for x in candidates if json.dumps(x['config'],sort_keys=True)==s));strategy=json.loads(chosen)
  models=dict(json.loads((V6/'selection.json').read_text())['models']);stage('final_refit',selected=selected)
  if selected in ('maxvit','dinol'):
   epochs=int(np.median([inner[(f,'inner',selected)]['best_epoch'] for f in range(3)]));models[selected]=train(0,'final',selected,note,final_end,epochs)
  elif selected=='expert':
   epochs=int(np.median([inner[(f,'inner',selected)]['best_epoch'] for f in range(3)]));folder=RUN/'models/final_fold0_expert_clean'
   args=[str(CODE/'v3_train.py'),'--v3-stage','final','--model','expert_person','--mode','tile','--fold','0','--out',str(folder),'--epochs','12','--schedule-epochs','12','--samples','400','--batch','2','--workers','2','--lr','.0003','--rare-fraction','.75','--loss','ce_dice','--seed','42','--max-seconds',str(max(120,int(final_end-time.monotonic()-30)))]
   core.command('final_expert_clean',args,note,final_end);ep=folder/'best.pt';hard=mine(0,'final',ep,note,final_end);models['expert']=train(0,'final','expert',note,final_end,epochs,ep,hard)
  frozen={'selected':selected,'strategy':strategy,'models':models,'development':comparison,'frozen_before_official_and_test':True,
   'checkpoint_sha256':{m['checkpoint']:sha(m['checkpoint']) for m in models.values()},'test_qualification':'previously observed; descriptive only'}
  atomic_json(RUN/'selection.json',frozen);atomic_json(RUN/'freeze_sha256.json',{'sha256':sha(RUN/'selection.json')});stage('frozen_final_reference',selected=selected)
  core.command('final_smoke',[str(CODE/'v7_predict.py'),'--selection',str(RUN/'selection.json'),'--out',str(RUN),'--deadline',str(DEADLINE),'--smoke'],note,eval_end)
  core.command('final_prediction',[str(CODE/'v7_predict.py'),'--selection',str(RUN/'selection.json'),'--out',str(RUN),'--deadline',str(DEADLINE)],note,eval_end)
  core.command('final_test',[str(CODE/'v3_public_test_offline.py'),'--freeze',str(RUN/'selection.json'),
   '--train-tar',str(V3/'public-train-validation.tar'),'--train-sha','50ac081a10e92e540720f94b6a5a021f1e84ff604a7668b5e95f9a7fbd5b222b',
   '--test-tar',str(V3/'public-test-labels.tar'),'--test-sha','f6a4c21b636740f78c66f71c850b785c2830a5eb10fd726320956bbcde6d0dda',
   '--v2',str(V6/'test_predictions/trainids'),'--v3',str(RUN/'test_predictions/trainids'),'--out',str(RUN/'public_test.json')],note,eval_end)
  test=json.loads((RUN/'public_test.json').read_text());reference=json.loads((V6/'report.json').read_text())
  if test['models']['v2']!=reference['test']['models']['v3']:raise ValueError('sixth-round reference replay differs')
  atomic_json(RUN/'report.json',{'status':'partial' if core.FAILURES else 'complete','selection':frozen,'validation':json.loads((RUN/'official_validation.json').read_text()),'test':test,'v6_reference':{'validation':reference['validation']['metrics'],'test':reference['test']['models']['v3']},'failures':core.FAILURES})
 for name in ('report.json','selection.json','cross_validation.json','error_analysis.json','pilot_decision.json'):
  shutil.copy2(RUN/name,Path(os.environ['YUN_RESULTS_DIR'])/name)
 stage('partial' if core.FAILURES else 'complete',selected=selected,failures=len(core.FAILURES))
 if core.FAILURES:raise RuntimeError('partial experiment; inspect report')

def main():
 RUN.mkdir(parents=True,exist_ok=False)
 atomic_json(RUN/'config.json',{'budget_seconds':BUDGET,'source_release':str(CODE),'pretrained_sha256':sha(ROOT/'cache/pretrained/dinov2_vitl14_pretrain.pth'),
  'pretrained_source':'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth','maxvit_pretrained_sha256':sha(ROOT/'cache/pretrained/maxvit_small_tf_512.pth'),
  'maxvit_pretrained_source':'https://huggingface.co/timm/maxvit_small_tf_512.in1k','pretraining_test_exposure':'unknown',
  'references':{'kaggle_cloud_3rd':'https://www.kaggle.com/competitions/understanding_cloud_organization/writeups/xuan-cao-finally-gm-1st-time-won-prize-money-and-3','kaggle_contrails_1st':'https://github.com/junkoda/kaggle_contrails_solution'},
  'baseline':BASE,'pilot':'4 epochs, 1 fold','test':'previously exposed; descriptive only','concurrency':2})
 note=None
 try:
  email=json.loads((core.PRIVATE/'preferences.json').read_text(encoding='utf-8'))['email']
  note=core.Client(direct=True).experiment('空天智能语义分割 UAVid 误差攻击 第7轮',heartbeat_timeout=300,heartbeat_interval=60,group='空天智能语义分割',email=email)
  note.__enter__();task_id=note.task['id'];fd=os.open(core.PRIVATE/('task-'+task_id+'.json'),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
  with os.fdopen(fd,'w',encoding='utf-8') as f:json.dump({'id':task_id,'task_key':note.task['task_key']},f)
  atomic_json(RUN/'notification_status.json',{'configured':True,'task_id':task_id});core.notify(note,'queued')
 except Exception as e:
  if note is not None and note.task is not None:core.close_note(note,type(e),e,e.__traceback__)
  note=None;atomic_json(RUN/'notification_status.json',{'configured':False,'error':type(e).__name__})
 stop=threading.Event();t=threading.Thread(target=core.monitor,args=(stop,),daemon=True);t.start()
 try:execute(note)
 except BaseException as e:
  atomic_json(RUN/'failures.json',core.FAILURES)
  if not (RUN/'report.json').exists():
   atomic_json(RUN/'report.json',{'status':'partial','fallback':str(V6/'selection.json'),'error':type(e).__name__+': '+str(e),'failures':core.FAILURES,'qualification':'existing sixth-round model retained'})
  for name in ('report.json','selection.json','cross_validation.json','error_analysis.json','pilot_decision.json'):
   if (RUN/name).exists(): shutil.copy2(RUN/name,Path(os.environ['YUN_RESULTS_DIR'])/name)
  stage('failed',error=type(e).__name__,message=str(e));core.close_note(note,type(e),e,e.__traceback__);raise
 else:core.close_note(note)
 finally:stop.set();t.join(15)
if __name__=='__main__':main()
