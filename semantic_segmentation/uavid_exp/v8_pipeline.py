"""Bounded eighth-round small-object and TTA experiment."""
from pathlib import Path
import hashlib,json,os,shutil,sys,threading,time,traceback
import numpy as np
import v2_pipeline as core
from v2_data import atomic_json
ROOT=Path('/data/aerospace-intelligence');V7=ROOT/'runs/uavid-20261106-v7-error-attack';V7FINAL=ROOT/'runs/uavid-20261106-v7-final-recovery';V6=ROOT/'runs/uavid-20261106-v6-pretrained';V3=ROOT/'runs/uavid-20261105-v3-error-attack';RUN=ROOT/'runs/uavid-20261106-v8-error-attack';CODE=Path(__file__).resolve().parent
START=time.monotonic();BUDGET=23*3600+45*60; DEADLINE=time.time()+BUDGET
core.RUN=RUN;core.START=START;core.BUDGET=BUDGET;core.ABS_DEADLINE=DEADLINE

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def stage(name,note=None,**kw):
 core.stage(name,**kw)
 if note is not None:
  try:note.metric(stage_index={'pilot':1,'inner':2,'outer':3,'final':4,'complete':5}.get(name,0),completed_folds=kw.get('fold',0))
  except Exception as e:print('endnote_metric_unavailable',type(e).__name__,flush=True)
def cmd(name,args,note,end):return core.command(name,[str(x) for x in args],note,end)
def expert(f,phase,note,end,epochs=8,init=None,hard=None,external=False):
 out=RUN/'models'/('%s_fold%d_%s_expert'%(phase,f,'external' if external else 'small'))
 if (out/'summary.json').exists() and (out/'best.pt').exists():return out/'best.pt'
 if init is None:init=V6/'models'/('%s_fold%d_expert'%(phase,f))/'best.pt'
 if hard is None:hard=V7/('hard_%s_%d.json'%(phase,f))
 args=[CODE/('v8_external_train.py' if external else 'v8_expert_train.py'),'--v3-stage',phase,'--v3-init',init,'--v3-hard',hard,'--model','expert_person','--mode','tile','--fold',f,'--out',out,'--epochs',epochs,'--schedule-epochs',8,'--samples',400,'--batch',2,'--workers',2,'--lr','.0001','--rare-fraction','.75','--loss','ce_dice','--seed',42,'--max-seconds',min(5400,max(120,int(end-time.monotonic()-30)))]
 if (out/'last.pt').exists():args+=['--resume']
 cmd('train_%s_%d'%(phase,f),args,note,end)
 summary=json.loads((out/'summary.json').read_text())
 if summary['epochs_completed']<epochs:raise RuntimeError('incomplete expert training')
 return out/'best.pt'
def evaluate(f,phase,ep,note,end,limit=None,tag=''):
 suffix=('_smoke' if limit else '')+tag;out=RUN/('eval_%s_%d%s.json'%(phase,f,suffix))
 if out.exists():return json.loads(out.read_text())
 args=[CODE/'v8_eval.py','--fold',f,'--stage',phase,'--out',out]
 if ep:args+=['--expert',ep]
 if limit:args+=['--limit',limit]
 cmd('eval_%s_%d%s'%(phase,f,suffix),args,note,end)
 return json.loads(out.read_text())
def metrics(row):return row['metrics']['miou'],row['metrics']['iou']['human']
def choose(inner):
 result=[]
 for x in inner:
  base=metrics(x['variants']['baseline']);candidates=[]
  for n,row in x['variants'].items():
   if n.startswith(('expert_','flip_expert_')):
    m,h=metrics(row)
    if m>base[0] and h>=base[1]:candidates.append((m,n))
  result.append(max(candidates)[1] if candidates else None)
 return result

def compare(rows,choices):
 names=['baseline','flip']+(['expert'] if all(choices) and all(choices[i] in rows[i]['variants'] for i in range(3)) else [])
 table={}
 for name in names:
  seq={}
  for i,row in enumerate(rows):
   key=choices[i] if name=='expert' else name
   for k,v in row['variants'][key]['sequence_confusions'].items():
    if k in seq:raise ValueError('outer sequence overlap')
    seq[k]=np.asarray(v,dtype=np.int64)
  if len(seq)!=20:raise ValueError('outer sequence coverage mismatch')
  m=sum(seq.values());ious=core.ious(m);table[name]={'miou':float(np.nanmean(ious)),'human_iou':float(ious[6]),'seq':seq}
 baseline=table['baseline'];out={}
 for name,row in table.items():
  keys=sorted(row['seq']);rng=np.random.default_rng(42);diff=[]
  for draw in rng.integers(0,len(keys),size=(2000,len(keys))):
   a=sum(row['seq'][keys[i]] for i in draw);b=sum(baseline['seq'][keys[i]] for i in draw)
   diff.append(float(np.nanmean(core.ious(a))-np.nanmean(core.ious(b))))
  dm=row['miou']-baseline['miou'];dh=row['human_iou']-baseline['human_iou'];ci=[float(x) for x in np.quantile(diff,[.025,.975])]
  out[name]={'miou':row['miou'],'human_iou':row['human_iou'],'delta_miou':dm,'delta_human':dh,'ci95':ci,'eligible':name!='baseline' and dm>=.002 and dh>=-.003 and ci[0]>0}
 winners=[name for name in out if out[name]['eligible']];chosen=max(winners,key=lambda n:out[n]['miou']) if winners else 'v7_fallback'
 return {'variants':out,'selected':chosen,'expert_inner_choices':choices,'qualification':'repeated historical development, not independent blind validation'}

def execute(note):
 end=START+BUDGET-20;pilot_end=START+4*3600;dev_end=START+17*3600;final_end=START+22*3600
 stage('pilot',note)
 external_ready=False
 try:
  
  audit=ROOT/'cache/aeroscapes-v8-audit.json';archive=ROOT/'cache/aeroscapes-v8-pilot.zip'
  if not audit.exists() or json.loads(audit.read_text()).get('archive_sha256')!=sha(archive):cmd('external_audit',[CODE/'v8_external_audit.py'],note,pilot_end)
  shutil.copy2(audit,RUN/'external_audit.json');external_ready=True
 except Exception as e:core.record_failure('external_audit',e)
 try:
  ep=expert(0,'inner',note,pilot_end);evaluate(0,'inner',ep,note,pilot_end,limit=1)
 except Exception as e:
  core.record_failure('pilot_expert',e);ep=None
  evaluate(0,'inner',None,note,pilot_end,limit=1)
 inner=[];have_expert=ep is not None;use_external=False
 for f in range(3):
  stage('inner',note,fold=f)
  try:
   ef=ep if f==0 else expert(f,'inner',note,dev_end,external=use_external)
  except Exception as e:core.record_failure('inner_expert_%d'%f,e);ef=None;have_expert=False
  try:
   row=evaluate(f,'inner',ef,note,dev_end)
   if f==0 and external_ready and ef is not None:
    try:
     ext=expert(0,'inner',note,dev_end,external=True);extrow=evaluate(0,'inner',ext,note,dev_end,tag='_external')
     def candidate_best(r):
      base=metrics(r['variants']['baseline']);vals=[metrics(v)[0] for k,v in r['variants'].items() if k.startswith(('expert_','flip_expert_')) and metrics(v)[1]>=base[1]];return max(vals) if vals else -1
     use_external=candidate_best(extrow)>=candidate_best(row)+.0005
     atomic_json(RUN/'external_decision.json',{'selected':use_external,'external_best':candidate_best(extrow),'uavid_best':candidate_best(row),'criterion':'inner-fold0 mIoU gain >=0.0005, human IoU non-decreasing against baseline','audit':str(RUN/'external_audit.json')})
     if use_external:row=extrow
    except Exception as e:core.record_failure('external_pilot',e);use_external=False
   inner.append(row)
  except Exception as e:
   core.record_failure('inner_eval_%d'%f,e);raise
 choices=choose(inner) if have_expert else [None]*3
 atomic_json(RUN/'inner_selection.json',{'expert_choices':choices,'gate':'each inner fold requires positive mIoU and nonnegative human IoU'})
 rows=[]
 for f in range(3):
  stage('outer',note,fold=f);ef=None
  if all(choices):
   try:ef=expert(f,'outer',note,dev_end,external=use_external)
   except Exception as e:core.record_failure('outer_expert_%d'%f,e);choices=[None]*3
  rows.append(evaluate(f,'outer',ef,note,dev_end))
 result=compare(rows,choices);atomic_json(RUN/'cross_validation.json',result);selected=result['selected']
 if selected=='v7_fallback':
  atomic_json(RUN/'report.json',{'status':'complete','selected':selected,'comparison':result,'fallback_report':str(V7FINAL/'report.json'),'failures':core.FAILURES})
 else:
  stage('final',note,selected=selected)
  source=json.loads((V7/'selection.json').read_text());models=source['models'];w=source['strategy']['weights'];strategy={'type':'v8','weights':w,'tta':selected=='flip','expert':None}
  if selected=='expert':
   most=max(set(choices),key=lambda x:(choices.count(x),x));strategy['tta']=most.startswith('flip_');strategy['threshold']=float(most.rsplit('_',1)[1]);strategy['expert']='expert8'
   clean=RUN/'models/final_expert_clean';args=[CODE/'v3_train.py','--v3-stage','final','--model','expert_person','--mode','tile','--fold',0,'--out',clean,'--epochs',8,'--schedule-epochs',8,'--samples',400,'--batch',2,'--workers',2,'--lr','.0003','--rare-fraction','.75','--loss','ce_dice','--seed',42,'--max-seconds',5400]
   cmd('final_expert_clean',args,note,final_end)
   hard=RUN/'hard_final_0.json';cmd('mine_final',[CODE/'v3_mine.py','--fold',0,'--stage','final','--checkpoint',clean/'best.pt','--out',hard],note,final_end)
   ep=expert(0,'final',note,final_end,init=clean/'best.pt',hard=hard,external=use_external);models['expert8']={'model':'expert_person','checkpoint':str(ep)}
  frozen={'selected':selected,'strategy':strategy,'models':models,'development':result,'frozen_before_official_and_test':True,'checkpoint_sha256':{m['checkpoint']:sha(m['checkpoint']) for m in models.values()},'test_qualification':'previously observed; descriptive only'}
  atomic_json(RUN/'selection.json',frozen);atomic_json(RUN/'freeze_sha256.json',{'sha256':sha(RUN/'selection.json')})
  for smoke in (True,False):
   args=[CODE/'v8_predict.py','--selection',RUN/'selection.json','--out',RUN,'--deadline',DEADLINE]
   if smoke:args+=['--smoke']
   cmd('final_smoke' if smoke else 'final_prediction',args,note,end)
  cmd('final_test',[CODE/'v3_public_test_offline.py','--freeze',RUN/'selection.json','--train-tar',V3/'public-train-validation.tar','--train-sha','50ac081a10e92e540720f94b6a5a021f1e84ff604a7668b5e95f9a7fbd5b222b','--test-tar',V3/'public-test-labels.tar','--test-sha','f6a4c21b636740f78c66f71c850b785c2830a5eb10fd726320956bbcde6d0dda','--v2',V7FINAL/'test_predictions/trainids','--v3',RUN/'test_predictions/trainids','--out',RUN/'public_test.json'],note,end)
  test=json.loads((RUN/'public_test.json').read_text());ref=json.loads((V7FINAL/'report.json').read_text())
  if test['models']['v2']!=ref['test']['models']['v3']:raise ValueError('seventh-round test reference replay mismatch')
  atomic_json(RUN/'report.json',{'status':'complete','selected':selected,'selection':frozen,'validation':json.loads((RUN/'official_validation.json').read_text()),'test':test,'v7_reference':ref['test']['models']['v3'],'failures':core.FAILURES})
 for n in ('report.json','cross_validation.json','inner_selection.json'):
  if (RUN/n).exists():shutil.copy2(RUN/n,Path(os.environ['YUN_RESULTS_DIR'])/n)
 stage('complete',note,selected=selected,failures=len(core.FAILURES))

def main():
 RUN.mkdir(parents=True,exist_ok=False);note=None
 atomic_json(RUN/'config.json',{'budget_seconds':BUDGET,'release':str(CODE),'target_miou_gain':.01,'promotion_miou_gain':.002,'max_human_iou_loss':.003,'sequence_bootstrap_lower_bound_positive':True,'test':'previously exposed, descriptive only','external_data':'AeroScapes train subset subject to overlap audit and paired pilot','new_pretrained_data':'none'})
 try:
  email=json.loads((core.PRIVATE/'preferences.json').read_text())['email'];note=core.Client(direct=True).experiment('空天智能 UAVid 第8轮小目标融合',heartbeat_timeout=300,heartbeat_interval=60,group='空天智能语义分割',email=email,notify_start=False);note.__enter__();tid=note.task['id'];fd=os.open(core.PRIVATE/('task-'+tid+'.json'),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
  with os.fdopen(fd,'w') as f:json.dump({'id':tid,'task_key':note.task['task_key']},f)
  atomic_json(RUN/'notification_status.json',{'configured':True,'task_id':tid});core.notify(note,'queued')
 except Exception as e:note=None;atomic_json(RUN/'notification_status.json',{'configured':False,'error':type(e).__name__});print('endnote_unavailable',type(e).__name__,flush=True)
 stop=threading.Event();thread=threading.Thread(target=core.monitor,args=(stop,),daemon=True);thread.start()
 try:execute(note)
 except BaseException as e:
  atomic_json(RUN/'failure.json',{'type':type(e).__name__,'message':str(e),'failures':core.FAILURES});core.stage('failed',error=type(e).__name__,message=str(e));core.close_note(note,type(e),e,e.__traceback__);raise
 else:core.close_note(note)
 finally:stop.set();thread.join(15)
if __name__=='__main__':main()
