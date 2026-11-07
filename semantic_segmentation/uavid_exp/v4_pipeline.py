"""Bounded human-recall fine-tuning from verified, sequence-disjoint checkpoints."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from collections import Counter
from pathlib import Path
import hashlib,json,os,shutil,threading,time
import numpy as np
import torch
import v2_pipeline as core
from v2_data import atomic_json

ROOT=Path('/data/aerospace-intelligence')
OLD=ROOT/'runs/uavid-20261105-v3-error-attack'
V2=ROOT/'runs/uavid-20261103-v2-error-attack'
RUN=ROOT/'runs/uavid-20261105-v4-human-recall'
CODE=Path(__file__).resolve().parent
START=time.monotonic(); BUDGET=8*3600
core.RUN,core.START,core.BUDGET=RUN,START,BUDGET
WEIGHTS={'sfa':2.,'conv':4.} # keys are v3_eval's candidate slots; both use course_resnet50

def stage(name,**values): core.stage(name,**values)
def checkpoint(fold,phase,key): return OLD/'models'/('%s_fold%d_%s'%(phase,fold,key))/'best.pt'
def verify_parent(fold,phase,key):
    file=checkpoint(fold,phase,key); obj=torch.load(file,map_location='cpu',weights_only=True)
    contract=json.loads((file.parent/'v3_contract.json').read_text())
    if obj['model_name']!='course_resnet50' or contract['fold']!=fold or contract['stage']!=phase:
        raise ValueError('invalid parent split or model')
    h=hashlib.sha256()
    with file.open('rb') as s:
        for block in iter(lambda:s.read(1024*1024),b''): h.update(block)
    return {'model':'course_resnet50','checkpoint':str(file),'sha256':h.hexdigest()}

def train(fold,phase,key,note,end,epochs=6,init=None):
    folder=RUN/'models'/('%s_fold%d_%s'%(phase,fold,key)); init=Path(init or checkpoint(fold,phase,'tile'))
    args=[str(CODE/'v4_train.py'),'--v3-stage',phase,'--human-weight',str(WEIGHTS[key]),'--v3-init',str(init),
          '--model','course_resnet50','--mode','tile','--fold',str(fold),'--out',str(folder),'--epochs',str(epochs),
          '--schedule-epochs','6','--samples','300','--batch','2','--workers','2','--lr','0.00005',
          '--rare-fraction','0.75','--loss','ce_dice','--seed','42','--max-seconds',str(max(120,int(end-time.monotonic()-20)))]
    if (folder/'last.pt').exists(): args.append('--resume')
    name='%s_fold%d_%s'%(phase,fold,key)
    for attempt in range(2):
        try:
            core.command(name,args,note,end);break
        except Exception:
            if attempt or time.monotonic()>end-180: raise
            if (folder/'last.pt').exists() and '--resume' not in args: args.append('--resume')
    summary=json.loads((folder/'summary.json').read_text());best=folder/'best.pt'
    if summary['epochs_completed']!=epochs: raise TimeoutError('incomplete epochs')
    epoch=int(torch.load(best,map_location='cpu',weights_only=True)['epoch'])
    return {'model':'course_resnet50','checkpoint':str(best),'best_epoch':epoch}

def many(jobs,note,end):
    out={}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(train,f,phase,k,note,end,ep,init):(f,phase,k) for f,phase,k,ep,init in jobs}
        for future in as_completed(futures):
            key=futures[future]
            try: out[key]=future.result()
            except Exception as e: core.record_failure(str(key),e)
    return out

def evaluate(fold,phase,models,choices,note,end):
    cfg=RUN/('eval_%s_%d_config.json'%(phase,fold));out=RUN/('eval_%s_%d.json'%(phase,fold))
    atomic_json(cfg,{'models':models,'choices':choices})
    core.command('eval_%s_%d'%(phase,fold),[str(CODE/'v3_eval.py'),'--fold',str(fold),'--stage',phase,'--config',str(cfg),'--out',str(out)],note,end)
    return json.loads(out.read_text())

def summarize(folds):
    names=set.intersection(*(set(f['variants']) for f in folds)); result={}
    for name in sorted(names):
        seq={}
        for fold in folds:
            for k,v in fold['variants'][name]['sequence_confusions'].items():
                if k in seq: raise ValueError('outer sequences overlap')
                seq[k]=np.asarray(v,dtype=np.int64)
        if len(seq)!=20: raise ValueError('missing outer sequence')
        m=sum(seq.values()); i=core.ious(m)
        result[name]={'miou':float(np.nanmean(i)),'human_iou':float(i[6]),'class_iou':[float(x) if np.isfinite(x) else None for x in i],
                      'confusion':m.tolist(),'sequence_confusions':{k:v.tolist() for k,v in seq.items()}}
    ref=result['previous_weighted']
    for name,v in result.items():
        v['delta_miou']=v['miou']-ref['miou'];v['delta_human_iou']=v['human_iou']-ref['human_iou']
        v['eligible']=name in WEIGHTS and v['delta_miou']>=.005 and v['delta_human_iou']>=-.02
    winners=[k for k,v in result.items() if v['eligible']]
    return {'variants':result,'selected':max(winners,key=lambda k:result[k]['miou']) if winners else 'v2_fallback',
            'note':'Repeated development on these folds; not an independent test. Public test labels were already observed and excluded.'}

def execute(note):
    end=START+7*3600; parents={}
    for f in range(3):
        for phase in ('inner','outer'):
            for key in ('global','tile'): parents['%s_%s_%s'%(f,phase,key)]=verify_parent(f,phase,key)
    atomic_json(RUN/'parents.json',parents)
    stage('inner_finetuning')
    jobs=[(f,'inner',k,6,None) for f in range(3) for k in WEIGHTS]
    inner=many(jobs,note,end)
    available=[k for k in WEIGHTS if all((f,'inner',k) in inner for f in range(3))]
    if not available: raise RuntimeError('no candidate completed all inner folds')
    evals=[]
    for f in range(3):
        models={k:parents['%s_inner_%s'%(f,k)] for k in ('global','tile')}
        models.update({k:inner[(f,'inner',k)] for k in available})
        evals.append(evaluate(f,'inner',models,{},note,end))
    available=[k for k in available if all(k in x['choices'] for x in evals)]
    stage('outer_finetuning',candidates=available)
    outer=many([(f,'outer',k,inner[(f,'inner',k)]['best_epoch'],None) for f in range(3) for k in available],note,end)
    available=[k for k in available if all((f,'outer',k) in outer for f in range(3))]
    folds=[]
    ref={'family':'weights','type':'mix','weights':{'global':.25,'tile':.75}}
    for f in range(3):
        models={k:parents['%s_outer_%s'%(f,k)] for k in ('global','tile')}
        models.update({k:outer[(f,'outer',k)] for k in available})
        choices={'previous_weighted':ref,**{k:evals[f]['choices'][k]['config'] for k in available}}
        folds.append(evaluate(f,'outer',models,choices,note,end))
    result=summarize(folds);atomic_json(RUN/'cross_validation.json',result)
    selected=result['selected'];selection={'selected':selected,'development':result,'test_labels_not_used':True}
    if selected!='v2_fallback':
        cfg=[x['choices'][selected]['config'] for x in evals]
        texts=[json.dumps(x,sort_keys=True) for x in cfg]; mode=Counter(texts).most_common(1)[0][0]
        chosen=json.loads(mode); epochs=int(np.median([inner[(f,'inner',selected)]['best_epoch'] for f in range(3)]))
        v2=json.loads((V2/'final_models.json').read_text()); init=v2['models']['tile']['checkpoint']
        stage('final_finetuning',selected=selected,epochs=epochs)
        model=train(0,'final',selected,note,START+BUDGET-60,epochs=epochs,init=init)
        selection.update({'strategy':chosen,'models':{'global':v2['models']['global'],'tile':v2['models']['tile'],selected:model},
                          'selected_slot_meaning':{'sfa':'human_weight_2','conv':'human_weight_4'}})
    else: selection['fallback']=str(V2/'final_models.json')
    atomic_json(RUN/'selection.json',selection)
    results=Path(os.environ['YUN_RESULTS_DIR']);shutil.copy2(RUN/'cross_validation.json',results/'cross_validation.json');shutil.copy2(RUN/'selection.json',results/'selection.json')
    stage('partial' if core.FAILURES else 'complete',selected=selected,failed=len(core.FAILURES))
    if core.FAILURES: raise RuntimeError('some candidates failed; partial results saved')

def main():
    RUN.mkdir(parents=True,exist_ok=False)
    atomic_json(RUN/'config.json',{'human_weights':WEIGHTS,'max_seconds':BUDGET,'concurrency':2,'source':str(CODE),
        'reference':'previous_weighted .25 global .75 tile','test_labels_excluded':True})
    note=None
    try:
        email=json.loads((core.PRIVATE/'preferences.json').read_text(encoding='utf-8'))['email']
        note=core.Client(direct=True).experiment('空天智能语义分割 UAVid 行人补强 第4轮',heartbeat_timeout=300,heartbeat_interval=60,group='空天智能语义分割',email=email)
        note.__enter__();task_id=note.task['id']; fd=os.open(core.PRIVATE/('task-'+task_id+'.json'),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w',encoding='utf-8') as f: json.dump({'id':task_id,'task_key':note.task['task_key']},f)
        atomic_json(RUN/'notification_status.json',{'configured':True,'task_id':task_id});core.notify(note,'queued')
    except Exception as e:
        if note is not None and note.task is not None: core.close_note(note,type(e),e,e.__traceback__)
        note=None;atomic_json(RUN/'notification_status.json',{'configured':False,'error':type(e).__name__})
    stop=threading.Event();t=threading.Thread(target=core.monitor,args=(stop,),daemon=True);t.start()
    try: execute(note)
    except BaseException as e:
        atomic_json(RUN/'failures.json',core.FAILURES);stage('failed',error=type(e).__name__,message=str(e));core.close_note(note,type(e),e,e.__traceback__);raise
    else: core.close_note(note)
    finally: stop.set();t.join(15)
if __name__=='__main__':main()
