"""Nested sequence CV, conservative model selection and one-shot test scoring."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from pathlib import Path
import hashlib, json, os, shutil, sys, threading, time
import numpy as np
import torch
import v2_pipeline as core
from v2_data import INDEX, atomic_json

RUN = Path('/data/aerospace-intelligence/runs/uavid-20261105-v3-error-attack')
V2 = Path('/data/aerospace-intelligence/runs/uavid-20261103-v2-error-attack')
CODE = Path(__file__).resolve().parent
START = time.monotonic()
BUDGET = 23*3600+45*60
DEADLINE = time.time()+BUDGET
core.RUN, core.START, core.BUDGET, core.ABS_DEADLINE = RUN, START, BUDGET, DEADLINE
SPECS = {
 'global':dict(model='course_resnet50',mode='global',lr=.00015,rare=0.,loss='ce',epochs=12,samples=300),
 'tile':dict(model='course_resnet50',mode='tile',lr=.00015,rare=.5,loss='ce_dice',epochs=16,samples=300),
 'sfa':dict(model='sfanet_resnet18',mode='tile',lr=.0003,rare=.5,loss='ce_dice',epochs=12,samples=300),
 'conv':dict(model='unet_convnext_tiny',mode='tile',lr=.00015,rare=.5,loss='ce_dice',epochs=12,samples=300),
 'expert_clean':dict(model='expert_person',mode='tile',lr=.0003,rare=.75,loss='ce_dice',epochs=12,samples=400),
 'expert_hard':dict(model='expert_person',mode='tile',lr=.0001,rare=.75,loss='ce_dice',epochs=3,samples=400)}

def stage(name, **kw): core.stage(name, **kw)
def fail(name, exc): core.record_failure(name,exc)
def cmd(name, args, note, end): core.command(name,args,note,end)
def train(key, fold, phase, note, end, epochs=None, hard=None, init=None):
    s=SPECS[key]; folder=RUN/'models'/('%s_fold%d_%s'%(phase,fold,key))
    ep=s['epochs'] if epochs is None else int(epochs)
    if ep < 1: raise ValueError('invalid epoch count')
    args=[str(CODE/'v3_train.py'),'--v3-stage',phase,'--model',s['model'],'--out',str(folder),'--fold',str(fold),
          '--mode',s['mode'],'--epochs',str(ep),'--schedule-epochs',str(s['epochs']),'--samples',str(s['samples']),
          '--batch','2','--workers','2','--lr',str(s['lr']),'--rare-fraction',str(s['rare']),'--loss',s['loss'],
          '--seed','42','--max-seconds',str(max(120,int(end-time.monotonic()-30)))]
    if hard: args+=['--v3-hard',str(hard)]
    if init: args+=['--v3-init',str(init)]
    if (folder/'last.pt').exists(): args+=['--resume']
    cmd('%s_fold%d_%s'%(phase,fold,key),args,note,end)
    summary=json.loads((folder/'summary.json').read_text())
    if summary['epochs_completed'] != ep: raise TimeoutError('training did not finish required epochs')
    ckpt=folder/'best.pt'
    saved=torch.load(ckpt,map_location='cpu',weights_only=True)
    return {'model':s['model'],'checkpoint':str(ckpt),'best_epoch':int(saved['epoch']),'path':str(folder)}

def many(jobs,note,end,concurrency):
    got={}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures={pool.submit(train,key,fold,phase,note,end,epochs,hard,init):(key,fold,phase) for key,fold,phase,epochs,hard,init in jobs}
        for f in as_completed(futures):
            key,fold,phase=futures[f]
            try: got[(key,fold,phase)]=f.result()
            except Exception as e: fail('%s_fold%d_%s'%(phase,fold,key),e)
    return got

def mine(fold,phase,expert,note,end):
    path=RUN/('hard_%s_fold%d.json'%(phase,fold))
    cmd('mine_%s_fold%d'%(phase,fold),[str(CODE/'v3_mine.py'),'--fold',str(fold),'--stage',phase,
        '--checkpoint',expert['checkpoint'],'--out',str(path)],note,end)
    return path

def evaluate(fold,phase,models,choices,note,end):
    config=RUN/('eval_%s_fold%d_config.json'%(phase,fold))
    out=RUN/('eval_%s_fold%d.json'%(phase,fold))
    atomic_json(config,{'models':models,'choices':choices})
    cmd('eval_%s_fold%d'%(phase,fold),[str(CODE/'v3_eval.py'),'--fold',str(fold),'--stage',phase,
        '--config',str(config),'--out',str(out)],note,end)
    return json.loads(out.read_text())

def summarize(folds):
    families=set.intersection(*(set(f['variants']) for f in folds))
    result={}
    for name in sorted(families):
        seq={}
        for fold in folds:
            for key,mat in fold['variants'][name]['sequence_confusions'].items():
                if key in seq: raise ValueError('overlapping outer sequences')
                seq[key]=np.asarray(mat,dtype=np.int64)
        if len(seq)!=20: raise ValueError('outer CV must cover 20 sequences')
        m=sum(seq.values()); i=core.ious(m)
        result[name]={'miou':float(np.nanmean(i)),'human_iou':float(i[6]),'class_iou':[float(x) if np.isfinite(x) else None for x in i],
                      'confusion':m.tolist(),'sequence_confusions':{k:v.tolist() for k,v in seq.items()}}
    base=result['baseline']
    for name,row in result.items():
        row['delta_miou']=row['miou']-base['miou']
        row['human_delta']=row['human_iou']-base['human_iou']
        row['eligible']=name!='baseline' and row['delta_miou']>=.005 and row['human_delta']>=-.02
    eligible=[x for x in result if result[x]['eligible']]
    selected=max(eligible,key=lambda x:result[x]['miou']) if eligible else 'v2_fallback'
    return {'variants':result,'selected_family':selected,'folds':3,'sequences':20,
            'qualification':'Outer folds select model family and are development evidence, not an independent test.'}

def execute(note):
    if not INDEX.exists(): raise RuntimeError('native cache missing')
    end=START+19*3600
    stage('inner_training')
    # Same jobs and sample counts; first/second epoch effects recorded as a caveat.
    pair=[('sfa',0,'inner',1,None,None),('conv',0,'inner',1,None,None)]
    t=time.monotonic(); par=many(pair,note,end,2); pt=time.monotonic()-t
    t=time.monotonic(); ser=many([(k,f,ph,2,h,i) for k,f,ph,e,h,i in pair if (k,f,ph) in par],note,end,1); st=time.monotonic()-t
    resources=[]
    if (RUN/'resources.jsonl').exists():
        resources=[json.loads(x)['gpu_memory_mib'] for x in (RUN/'resources.jsonl').read_text().splitlines()]
    peak=max(resources or [0]); concurrency=2 if len(par)==len(ser)==2 and pt*1.10<st and peak<20480 else 1
    atomic_json(RUN/'efficiency.json',{'parallel_seconds':pt,'serial_seconds':st,'peak_gpu_mib':peak,'concurrency':concurrency,
       'caveat':'first versus second epoch; same jobs and sample counts, exploratory comparison'})
    stage('throughput_checked',concurrency=concurrency,parallel_seconds=round(pt),serial_seconds=round(st),peak_gpu_mib=peak)
    jobs=[(k,f,'inner',None,None,None) for f in range(3) for k in ('global','tile','sfa','conv','expert_clean')]
    inner=many(jobs,note,end,concurrency)
    if not all((k,f,'inner') in inner for f in range(3) for k in ('global','tile')):
        raise RuntimeError('nested baseline is incomplete')
    for f in range(3):
        expert=inner.get(('expert_clean',f,'inner'))
        if expert:
            try:
                hard=mine(f,'inner',expert,note,end)
                inner[('expert_hard',f,'inner')]=train('expert_hard',f,'inner',note,end,hard=hard,init=expert['checkpoint'])
            except Exception as e: fail('inner_hard_fold%d'%f,e)
    choices=[]; inner_eval=[]
    for f in range(3):
        models={k:inner[(k,f,'inner')] for k in SPECS if (k,f,'inner') in inner}
        ev=evaluate(f,'inner',models,{},note,end); inner_eval.append(ev)
        choices.append({k:v['config'] for k,v in ev['choices'].items()})
    stage('outer_refit_and_evaluation')
    outer=[]
    for f in range(3):
        needed={'global','tile'}
        for family,cfg in choices[f].items():
            needed.update(k for k,v in cfg.get('weights',{}).items() if v)
            if cfg['type']=='expert': needed.add(cfg['expert'])
        if 'expert_hard' in needed: needed.add('expert_clean')
        models={}
        for key in sorted(needed-{'expert_hard'}):
            try: models[key]=train(key,f,'outer',note,end,epochs=inner[(key,f,'inner')]['best_epoch'])
            except Exception as e: fail('outer_fold%d_%s'%(f,key),e)
        if 'expert_hard' in needed and 'expert_clean' in models:
            try:
                hard=mine(f,'outer',models['expert_clean'],note,end)
                models['expert_hard']=train('expert_hard',f,'outer',note,end,epochs=inner[('expert_hard',f,'inner')]['best_epoch'],hard=hard,init=models['expert_clean']['checkpoint'])
            except Exception as e: fail('outer_hard_fold%d'%f,e)
        if not {'global','tile'}.issubset(models): raise RuntimeError('outer baseline incomplete')
        good={}
        for family,cfg in choices[f].items():
            req={k for k,v in cfg.get('weights',{}).items() if v}
            if cfg['type']=='expert': req.add(cfg['expert'])
            if req.issubset(models): good[family]=cfg
        outer.append(evaluate(f,'outer',models,good,note,end))
    comparison=summarize(outer); atomic_json(RUN/'cross_validation.json',comparison)
    family=comparison['selected_family']
    core.notify(note,'metric',cv_miou=comparison['variants'].get(family,comparison['variants']['baseline'])['miou'])
    if family=='v2_fallback':
        selection={'family':family,'reason':'no nested-CV candidate satisfies +0.005 mIoU and human IoU >= baseline-0.02',
                   'v2_final':str(V2/'final_models.json'),'comparison':comparison}
        predroot=V2/'test_predictions/trainids'
        shutil.copy2(V2/'official_validation.json',RUN/'official_validation.json')
        shutil.copy2(V2/'prediction_summary.json',RUN/'prediction_summary.json')
    else:
        configs=[inner_eval[f]['choices'][family] for f in range(3)]
        enc=[json.dumps(c['config'],sort_keys=True) for c in configs]
        freq=Counter(enc); maxfreq=max(freq.values())
        best=max((v for v in freq if freq[v]==maxfreq),key=lambda v:sum(c['inner_gain'] for c in configs if json.dumps(c['config'],sort_keys=True)==v))
        strategy=json.loads(best); needed={'global','tile'}
        needed.update(k for k,v in strategy.get('weights',{}).items() if v)
        if strategy['type']=='expert': needed.add(strategy['expert'])
        if 'expert_hard' in needed: needed.add('expert_clean')
        stage('final_refit',family=family)
        models={}
        for key in sorted(needed-{'expert_hard'}):
            epochs=int(np.median([inner[(key,f,'inner')]['best_epoch'] for f in range(3) if (key,f,'inner') in inner]))
            models[key]=train(key,0,'final',note,START+BUDGET-2*3600,epochs=epochs)
        if 'expert_hard' in needed:
            hard=mine(0,'final',models['expert_clean'],note,START+BUDGET-2*3600)
            epochs=int(np.median([inner[('expert_hard',f,'inner')]['best_epoch'] for f in range(3) if ('expert_hard',f,'inner') in inner]))
            models['expert_hard']=train('expert_hard',0,'final',note,START+BUDGET-2*3600,epochs=epochs,hard=hard,init=models['expert_clean']['checkpoint'])
        selection={'family':family,'strategy':strategy,'models':models,'comparison':comparison}
        predroot=RUN/'test_predictions/trainids'
    paths = ([Path(v['checkpoint']) for v in selection.get('models',{}).values()] if family != 'v2_fallback' else
             [Path(v['checkpoint']) for v in json.loads((V2/'final_models.json').read_text())['models'].values()])
    selection['checkpoint_sha256'] = {}
    for path in paths:
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''): digest.update(chunk)
        selection['checkpoint_sha256'][str(path)]=digest.hexdigest()
    selection['frozen_before_official_and_test']=True
    atomic_json(RUN/'selection_before_test.json',selection)
    freezehash=hashlib.sha256((RUN/'selection_before_test.json').read_bytes()).hexdigest()
    atomic_json(RUN/'freeze_sha256.json',{'sha256':freezehash,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    stage('selection_frozen',family=family,selection_sha256=freezehash)
    if family!='v2_fallback':
        cmd('final_prediction',[str(CODE/'v3_predict.py'),'--selection',str(RUN/'selection_before_test.json'),
            '--out',str(RUN),'--deadline',str(DEADLINE-10)],note,START+BUDGET-10)
    official=json.loads((RUN/'official_validation.json').read_text())
    try:
        stage('public_test_once')
        cmd('public_test',[str(CODE/'v3_public_test.py'),'--freeze',str(RUN/'selection_before_test.json'),
            '--v2',str(V2/'test_predictions/trainids'),'--v3',str(predroot),'--out',str(RUN/'public_test.json')],note,START+BUDGET-10)
        public=json.loads((RUN/'public_test.json').read_text())
    except Exception as e:
        fail('public_test',e); public={'status':'unavailable','reason':str(e),'log':str(RUN/'logs/public_test.log')}
    report={'status':'partial' if core.FAILURES else 'complete','family':family,'cv':comparison,
            'official_validation':official,'public_test':public,'prediction_directory':str(predroot),
            'failures':core.FAILURES,'budget_seconds':BUDGET,'elapsed_seconds':round(time.monotonic()-START)}
    atomic_json(RUN/'report.json',report)
    results=Path(os.environ['YUN_RESULTS_DIR'])
    for name in ('report.json','cross_validation.json','selection_before_test.json','official_validation.json','freeze_sha256.json'):
        shutil.copy2(RUN/name,results/name)
    if (RUN/'public_test.json').exists(): shutil.copy2(RUN/'public_test.json',results/'public_test.json')
    core.notify(note,'metric',official_val_miou=official['metrics']['miou'])
    stage('complete' if not core.FAILURES else 'partial',status=report['status'],family=family)
    if core.FAILURES: raise RuntimeError('partial results; inspect report and logs')

def rescue(note, error):
    # A failed or time-limited development run still preserves the established model.
    if (RUN/'selection_before_test.json').exists() or time.monotonic() >= START+BUDGET-120: return
    old=json.loads((V2/'final_models.json').read_text())
    selection={'family':'v2_fallback','reason':'development incomplete: '+str(error),
               'v2_final':str(V2/'final_models.json'),'frozen_before_official_and_test':True,'checkpoint_sha256':{}}
    for item in old['models'].values():
        path=Path(item['checkpoint']); digest=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''): digest.update(chunk)
        selection['checkpoint_sha256'][str(path)]=digest.hexdigest()
    atomic_json(RUN/'selection_before_test.json',selection)
    atomic_json(RUN/'freeze_sha256.json',{'sha256':hashlib.sha256((RUN/'selection_before_test.json').read_bytes()).hexdigest()})
    shutil.copy2(V2/'official_validation.json',RUN/'official_validation.json')
    try:
        cmd('public_test_rescue',[str(CODE/'v3_public_test.py'),'--freeze',str(RUN/'selection_before_test.json'),
            '--v2',str(V2/'test_predictions/trainids'),'--v3',str(V2/'test_predictions/trainids'),
            '--out',str(RUN/'public_test.json')],note,START+BUDGET-10)
        public=json.loads((RUN/'public_test.json').read_text())
    except Exception as e:
        fail('public_test_rescue',e); public={'status':'unavailable','reason':str(e)}
    atomic_json(RUN/'report.json',{'status':'partial','family':'v2_fallback','error':str(error),
        'official_validation':json.loads((RUN/'official_validation.json').read_text()),'public_test':public,
        'prediction_directory':str(V2/'test_predictions/trainids'),'failures':core.FAILURES})
    results=Path(os.environ['YUN_RESULTS_DIR'])
    for name in ('report.json','selection_before_test.json','freeze_sha256.json','official_validation.json'):
        shutil.copy2(RUN/name,results/name)

def main():
    RUN.mkdir(parents=True,exist_ok=False)
    atomic_json(RUN/'config.json',{'specs':SPECS,'budget_seconds':BUDGET,'seed':42,'source_release':str(CODE)})
    note=None
    try:
        email=json.loads((core.PRIVATE/'preferences.json').read_text(encoding='utf-8'))['email']
        note=core.Client(direct=True).experiment('空天智能语义分割 UAVid 第3轮 嵌套验证',heartbeat_timeout=300,heartbeat_interval=60,group='空天智能语义分割',email=email)
        note.__enter__(); task_id=note.task['id']
        fd=os.open(core.PRIVATE/('task-'+task_id+'.json'),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w',encoding='utf-8') as f: json.dump({'id':task_id,'task_key':note.task['task_key']},f)
        atomic_json(RUN/'notification_status.json',{'configured':True,'task_id':task_id})
        core.notify(note,'queued')
    except Exception as e:
        if note is not None and note.task is not None: core.close_note(note,type(e),e,e.__traceback__)
        note=None; atomic_json(RUN/'notification_status.json',{'configured':False,'error':type(e).__name__})
    stop=threading.Event(); monitor=threading.Thread(target=core.monitor,args=(stop,),daemon=True); monitor.start()
    try: execute(note)
    except BaseException as e:
        try: rescue(note,e)
        except Exception as rescue_error: fail('fallback',rescue_error)
        atomic_json(RUN/'failures.json',core.FAILURES)
        if not (RUN/'report.json').exists(): stage('failed',error=type(e).__name__,message=str(e))
        core.close_note(note,type(e),e,e.__traceback__)
        raise
    else: core.close_note(note)
    finally: stop.set(); monitor.join(15)
if __name__=='__main__': main()
