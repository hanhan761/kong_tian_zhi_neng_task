"""Bounded UAVid model selection, expert gating, and held-out evaluation."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import threading
import time

import torch
from endnote_client import Client, NotificationError
from ensemble import cache_probabilities, select_ensemble, choose_expert, evaluate_official, export_test_predictions

PROJECT = Path('/data/aerospace-intelligence')
RUN = PROJECT/'runs/uavid-20261103-v1'
PRIVATE = Path.home()/'.config/endnote/aerospace-intelligence'
CODE = Path(__file__).resolve().parent
PYTHON = sys.executable
DUE_SECONDS = 48*3600
CASES = [
    {'id':'course_mobile_sgd','model':'course_mobile','optimizer':'sgd','lr':0.1,'batch':2},
    {'id':'course_mobile_adamw','model':'course_mobile','optimizer':'adamw','lr':0.0003,'batch':2},
    {'id':'course_resnet50_adamw','model':'course_resnet50','optimizer':'adamw','lr':0.0003,'batch':2},
    {'id':'unet_resnet34','model':'unet_resnet34','optimizer':'adamw','lr':0.0003,'batch':2},
    {'id':'fpn_resnet34','model':'fpn_resnet34','optimizer':'adamw','lr':0.0003,'batch':2},
    {'id':'segformer_mitb0','model':'segformer_mitb0','optimizer':'adamw','lr':0.00006,'batch':2},
    {'id':'unetformer_resnet18','model':'unetformer_resnet18','optimizer':'adamw','lr':0.0006,'batch':2},
]
EXPERT = {'id':'expert_mobile_unet','model':'expert_mobile_unet','optimizer':'adamw','lr':0.0003,'batch':8}
active_pids = set()
active_lock = threading.Lock()
note_lock = threading.Lock()
note_started = False
failures = []
deadline = time.monotonic()+DUE_SECONDS


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp,path)


def state(stage, **details):
    value = {'stage':stage,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**details}
    atomic_json(RUN/'state.json',value)
    print(json.dumps(value,ensure_ascii=False),flush=True)


def notify(note, method, **metrics):
    if note is None:
        return
    try:
        with note_lock:
            getattr(note, method)(**metrics)
    except Exception as exc:
        print('endnote_event_error=' + type(exc).__name__,flush=True)


def ensure_started(note):
    global note_started
    with note_lock:
        if note is not None and not note_started:
            try:
                note.started()
                note_started = True
            except Exception as exc:
                print('endnote_started_error=' + type(exc).__name__,flush=True)


def gpu_processes():
    command = ['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits']
    r = subprocess.run(command,text=True,capture_output=True,timeout=10,check=True)
    return {int(line.strip()) for line in r.stdout.splitlines() if line.strip().isdigit()}


def wait_for_gpu(note):
    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError('pipeline budget expired while waiting for GPU')
        with active_lock:
            own = set(active_pids) | {os.getpid()}
        foreign = gpu_processes() - own
        if not foreign:
            return
        print('gpu_waiting_for_other_workload',flush=True)
        time.sleep(30)


def monitor_resources(stop):
    log = RUN/'resources.jsonl'
    while not stop.wait(10):
        try:
            r = subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,temperature.gpu',
                                '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=8,check=True)
            mem, util, temp = [int(x.strip()) for x in r.stdout.splitlines()[0].split(',')]
            row = {'time':time.time(),'memory_mib':mem,'utilization_pct':util,'temperature_c':temp}
            with log.open('a',encoding='utf-8') as f:
                f.write(json.dumps(row)+'\n')
        except Exception:
            pass


def disk_check():
    if shutil.disk_usage(PROJECT).free < 50*1024**3:
        raise RuntimeError('project disk reserve below 50 GiB')


def train_case(case, epochs, note, train_all=False):
    disk_check()
    wait_for_gpu(note)
    folder = RUN/'models'/case['id']
    folder.mkdir(parents=True,exist_ok=True)
    resume = (folder/'last.pt').exists()
    command = [PYTHON,'-u',str(CODE/'train.py'),'--model',case['model'],'--run-dir',str(folder),
               '--lr',str(case['lr']),'--optimizer',case['optimizer'],'--epochs',str(epochs),
               '--schedule-epochs','30','--batch-size',str(case['batch']),'--workers','4','--seed','42']
    if resume:
        command.append('--resume')
    if train_all:
        command.append('--train-all')
    started = time.monotonic()
    with (folder/'process.log').open('a',encoding='utf-8') as log:
        proc = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        with active_lock:
            active_pids.add(proc.pid)
        ensure_started(note)
        try:
            while proc.poll() is None:
                if time.monotonic() >= deadline or time.monotonic()-started > 8*3600:
                    proc.terminate()
                    try: proc.wait(timeout=20)
                    except subprocess.TimeoutExpired: proc.kill(); proc.wait()
                    raise TimeoutError('training stage exceeded its limit')
                time.sleep(3)
            if proc.returncode != 0:
                raise RuntimeError('training exited with code %s; inspect process.log' % proc.returncode)
        finally:
            with active_lock:
                active_pids.discard(proc.pid)
    summary = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    result = dict(case,path=str(folder),summary=summary,wall_seconds=round(time.monotonic()-started,2))
    print(json.dumps({'stage':'model_completed','id':case['id'],'epochs':epochs,
                      'dev_miou':summary['best_dev_miou'],'wall_seconds':result['wall_seconds']}),flush=True)
    return result


def run_batch(cases, epochs, concurrency, note, train_all=False):
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(train_case,c,epochs,note,train_all):c for c in cases}
        for future in as_completed(futures):
            case = futures[future]
            try:
                result = future.result()
                results.append(result)
                value = result['summary']['best_dev_miou']
                if value is not None:
                    notify(note,'metric',dev_miou=float(value),epoch=int(epochs))
            except Exception as exc:
                failures.append({'id':case['id'],'stage':'train','error':type(exc).__name__+': '+str(exc)})
                print(json.dumps({'stage':'model_failed','id':case['id'],'error':type(exc).__name__}),flush=True)
    return results


def read_best_epoch(case):
    d = torch.load(Path(case['path'])/'best.pt',map_location='cpu',weights_only=True)
    return int(d['epoch'])


def best_distinct(cases, number):
    chosen, seen = [], set()
    for case in sorted(cases,key=lambda x:x['summary']['best_dev_miou'],reverse=True):
        if case['model'] not in seen:
            chosen.append(case)
            seen.add(case['model'])
        if len(chosen) >= number:
            break
    return chosen


def execute(note):
    disk_check()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA not available on authorized target')
    # The two pilot models were independently sampled below 2 GiB each.
    # Compare two real first epochs in parallel with their second epochs in series.
    a,b = CASES[0], CASES[3]
    state('parallel_pilot',models=[a['id'],b['id']])
    t = time.monotonic()
    pilot = run_batch([a,b],1,2,note)
    parallel_seconds = time.monotonic()-t
    t = time.monotonic()
    serial = []
    for case in [a,b]:
        if any(x['id']==case['id'] for x in pilot):
            serial.extend(run_batch([case],2,1,note))
    serial_seconds = time.monotonic()-t
    concurrency = 2 if len(pilot)==2 and len(serial)==2 and parallel_seconds*1.05 < serial_seconds else 1
    efficiency = {'parallel_two_epochs_seconds':round(parallel_seconds,2),
                  'serial_two_epochs_seconds':round(serial_seconds,2),
                  'chosen_concurrency':concurrency}
    atomic_json(RUN/'efficiency.json',efficiency)
    state('baseline_training',efficiency=efficiency)
    pilot_ok = {x['id'] for x in serial}
    remaining = [c for c in CASES if c['id'] not in (a['id'],b['id']) or c['id'] in pilot_ok]
    initial = run_batch(remaining+[EXPERT],12,concurrency,note)
    expert = next((x for x in initial if x['id']==EXPERT['id']),None)
    trained = [x for x in initial if x['id'] != EXPERT['id']]
    if len(trained)<2 or expert is None:
        raise RuntimeError('not enough successful baselines or expert training failed')
    # A bounded learning-rate perturbation for two distinct best architectures.
    tuned_specs = []
    for c in best_distinct(trained,2):
        tuned_specs.append({'id':c['id']+'_lrhalf','model':c['model'],'optimizer':c['optimizer'],
                            'lr':c['lr']/2,'batch':c['batch']})
    state('hyperparameter_search',models=[x['id'] for x in tuned_specs])
    tuned = run_batch(tuned_specs,12,concurrency,note)
    trained.extend(tuned)
    # Give the two best development architectures a longer budget.
    extend_specs = [{k:c[k] for k in ('id','model','optimizer','lr','batch')} for c in best_distinct(trained,2)]
    state('extend_top_models',models=[x['id'] for x in extend_specs])
    extended = run_batch(extend_specs,30,concurrency,note)
    for x in extended:
        trained = [c for c in trained if c['id'] != x['id']] + [x]
    ranked = best_distinct(trained,3)
    if len(ranked)<2:
        raise RuntimeError('at least two distinct baseline architectures are needed for fusion')
    state('development_predictions',models=[x['id'] for x in ranked])
    for candidate in ranked:
        cache_probabilities(candidate,'dev',RUN)
    selected, ensemble_scores = select_ensemble(ranked,RUN)
    state('expert_gate_selection',models=[x['id'] for x in selected])
    threshold, expert_scores = choose_expert(selected,expert,RUN)
    selection = {'dev_ranking':[{'id':x['id'],'model':x['model'],'best_dev_miou':x['summary']['best_dev_miou']} for x in sorted(trained,key=lambda z:z['summary']['best_dev_miou'],reverse=True)],
                 'ensemble_trials':ensemble_scores,'expert_gate_trials':expert_scores,
                 'selected_models':[x['id'] for x in selected], 'expert_threshold':threshold,
                 'selection_basis':'training-sequence development split; official validation not used'}
    atomic_json(RUN/'selection_before_official_val.json',selection)
    # Refit chosen configurations on all 200 training frames for their development-selected epoch count.
    final_models = []
    state('refit_on_all_training_sequences',models=selection['selected_models'])
    for c in selected:
        spec = {'id':'final_'+c['id'],'model':c['model'],'optimizer':c['optimizer'],'lr':c['lr'],'batch':c['batch']}
        result = run_batch([spec],read_best_epoch(c),1,note,train_all=True)
        if result:
            final_models.extend(result)
        else:
            final_models.append(c)
    final_expert = expert
    if threshold is not None:
        spec = {'id':'final_expert_mobile_unet','model':EXPERT['model'],'optimizer':EXPERT['optimizer'],
                'lr':EXPERT['lr'],'batch':EXPERT['batch']}
        result = run_batch([spec],read_best_epoch(expert),1,note,train_all=True)
        if result:
            final_expert = result[0]
    # All decisions are fixed in selection_before_official_val.json before this stage.
    evaluation_candidates = trained + [x for x in final_models if x['id'] not in {c['id'] for c in trained}]
    state('official_validation',candidate_count=len(evaluation_candidates))
    for c in evaluation_candidates:
        cache_probabilities(c,'official_val',RUN)
    official = evaluate_official(evaluation_candidates,final_models,final_expert,threshold,RUN)
    notify(note,'metric',official_val_miou=float(official['selected_final']['miou']))
    state('export_unlabelled_test_predictions')
    predictions = export_test_predictions(final_models,final_expert,threshold,RUN)
    summary = {'status':'partial' if failures else 'complete', 'selection':selection,
               'official_validation':official, 'test_predictions':predictions, 'failures':failures,
               'efficiency':efficiency, 'classes':['clutter','building','road','static_car','tree','vegetation','human','moving_car'],
               'evaluation':'full-resolution masks; original official validation sequences excluded from tuning',
               'training_split':'baseline development: 160 train + 40 dev; final selected models refit on 200 training frames',
               'historical_checkpoint_note':'course-provided UAVid checkpoints not used for selection because their validation history is unknown'}
    atomic_json(RUN/'report.json',summary)
    result_dir = Path(os.environ['YUN_RESULTS_DIR'])
    shutil.copy2(RUN/'report.json',result_dir/'report.json')
    shutil.copy2(RUN/'selection_before_official_val.json',result_dir/'selection.json')
    state('complete',status=summary['status'],official_miou=official['selected_final']['miou'],test_images=predictions['images'])
    if failures:
        raise RuntimeError('some requested model runs failed; see report.json')
    return summary


def main():
    RUN.mkdir(parents=True,exist_ok=False)
    atomic_json(RUN/'experiment_config.json',{'candidates':CASES,'expert':EXPERT,'dev_sequences':['seq4','seq9','seq14','seq35'],
                'seed':42,'pilot_epochs':1,'initial_epochs':12,'extended_epochs':30,'deadline_seconds':DUE_SECONDS})
    note = None
    try:
        email = json.loads((PRIVATE/'preferences.json').read_text(encoding='utf-8'))['email']
        note = Client(direct=True).experiment('空天智能语义分割 UAVid 多基线融合专家 第1轮',
                                              heartbeat_timeout=300,heartbeat_interval=60,
                                              group='空天智能语义分割',email=email)
        note.__enter__()
        task_id = note.task['id']
        private_file = PRIVATE/('task-'+task_id+'.json')
        descriptor = os.open(private_file,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(descriptor,'w',encoding='utf-8') as f:
            json.dump({'id':task_id,'task_key':note.task['task_key']},f)
        print('endnote_task_id='+task_id,flush=True)
        notify(note,'queued')
    except Exception as exc:
        if note is not None and note.task is not None:
            try:
                note.__exit__(type(exc),exc,exc.__traceback__)
            except Exception:
                pass
        note = None
        print('endnote_setup_error='+type(exc).__name__,flush=True)
    stop = threading.Event()
    monitor = threading.Thread(target=monitor_resources,args=(stop,),daemon=True)
    monitor.start()
    try:
        execute(note)
    except BaseException as exc:
        state('failed',error=type(exc).__name__,failure_count=len(failures))
        atomic_json(RUN/'failures.json',failures)
        if note is not None:
            note.__exit__(type(exc),exc,exc.__traceback__)
        raise
    else:
        if note is not None:
            note.__exit__(None,None,None)
    finally:
        stop.set()
        monitor.join(15)

if __name__ == '__main__':
    main()
