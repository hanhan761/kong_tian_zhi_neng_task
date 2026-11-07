"""Bounded three-fold error-focused UAVid experiment on the authorized GPU."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
from endnote_client import Client
from v2_data import INDEX, atomic_json

PROJECT = Path('/data/aerospace-intelligence')
RUN = PROJECT / 'runs/uavid-20261103-v2-error-attack'
CODE = Path(__file__).resolve().parent
PRIVATE = Path.home() / '.config/endnote/aerospace-intelligence'
PYTHON = sys.executable
# Leave 15 minutes for the already completed preparation and GPU smoke checks.
BUDGET = 23 * 3600 + 45 * 60
START = time.monotonic()
ABS_DEADLINE = time.time() + BUDGET
ACTIVE = set()
LOCK = threading.Lock()
NOTE_LOCK = threading.Lock()
NOTE_STARTED = False
FAILURES = []

COURSE = dict(id='course_global', model='course_resnet50', mode='global', lr=.00015,
              rare=0., loss='ce', epochs=20, samples=300, seconds=2700)
FPN = dict(id='fpn_global', model='fpn_resnet34', mode='global', lr=.0003,
           rare=0., loss='ce', epochs=20, samples=300, seconds=2700)
PILOTS = [
    dict(id='course_tile_uniform', model='course_resnet50', mode='tile', lr=.00015, rare=0., loss='ce'),
    dict(id='course_tile_rare', model='course_resnet50', mode='tile', lr=.00015, rare=.5, loss='ce'),
    dict(id='course_tile_rare_dice', model='course_resnet50', mode='tile', lr=.00015, rare=.5, loss='ce_dice'),
    dict(id='course_tile_rare_lovasz', model='course_resnet50', mode='tile', lr=.00015, rare=.5, loss='ce_lovasz'),
    dict(id='unetformer_tile_rare_dice', model='unetformer_resnet18', mode='tile', lr=.0003, rare=.5, loss='ce_dice'),
]
for candidate in PILOTS:
    candidate.update(epochs=16, samples=300, seconds=5400)
EXPERT = dict(id='expert_person', model='expert_person', mode='tile', lr=.0003,
              rare=.75, loss='ce_dice', epochs=12, samples=400, seconds=3600)
GATES = {'expert_080_005': [0.80, 0.05], 'expert_090_002': [0.90, 0.02],
         'expert_095_001': [0.95, 0.01]}


def stage(name, **values):
    obj = {'stage': name, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'elapsed_seconds': round(time.monotonic()-START), **values}
    atomic_json(RUN / 'state.json', obj)
    print(json.dumps(obj, ensure_ascii=False, allow_nan=False), flush=True)


def notify(note, method, **values):
    if note is not None:
        try:
            with NOTE_LOCK:
                getattr(note, method)(**values)
        except Exception as exc:
            print('endnote_event_error=' + type(exc).__name__, flush=True)


def close_note(note, kind=None, error=None, tb=None):
    if note is not None:
        try:
            note.__exit__(kind, error, tb)
        except Exception as exc:
            print('endnote_finish_error=' + type(exc).__name__, flush=True)


def gpu_pids():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, check=True, timeout=10)
    return {int(x.strip()) for x in result.stdout.splitlines() if x.strip().isdigit()}


def ensure_start(note):
    global NOTE_STARTED
    with NOTE_LOCK:
        if note is not None and not NOTE_STARTED:
            try:
                note.started()
                NOTE_STARTED = True
            except Exception as exc:
                print('endnote_started_error=' + type(exc).__name__, flush=True)


def command(label, args, note, until=None):
    if shutil.disk_usage(PROJECT).free < 30 * 1024**3:
        raise RuntimeError('project disk reserve below 30 GiB')
    end = min(START+BUDGET-10, until if until is not None else START+BUDGET-10)
    log = RUN / 'logs' / (label + '.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    while True:
        if time.monotonic() >= end:
            raise TimeoutError('stage budget expired before ' + label)
        with LOCK:
            foreign = gpu_pids() - ACTIVE - {os.getpid()}
            if not foreign:
                stream = log.open('a', encoding='utf-8')
                proc = subprocess.Popen([PYTHON, '-u', *args], stdout=stream, stderr=subprocess.STDOUT,
                                        env={**os.environ, 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '2',
                                             'PYTHONDONTWRITEBYTECODE': '1'})
                ACTIVE.add(proc.pid)
                break
        print('gpu_waiting_for_other_workload', flush=True)
        time.sleep(20)
    ensure_start(note)
    try:
        while proc.poll() is None:
            if time.monotonic() >= end:
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                raise TimeoutError('stage budget exceeded: ' + label)
            time.sleep(3)
        if proc.returncode:
            raise RuntimeError('%s exited %s; inspect its project log' % (label, proc.returncode))
    finally:
        stream.close()
        with LOCK:
            ACTIVE.discard(proc.pid)
    return log


def train_case(spec, fold, note, phase_end, train_all=False, epochs=None, label=None):
    name = label or ('%s_fold%d' % (spec['id'], fold))
    folder = RUN / 'models' / name
    remaining = phase_end - time.monotonic()
    if remaining < 120:
        raise TimeoutError('insufficient phase time for ' + name)
    epoch_count = epochs if epochs is not None else spec['epochs']
    args = [str(CODE/'v2_train.py'), '--model', spec['model'], '--out', str(folder), '--fold', str(fold),
            '--mode', spec['mode'], '--epochs', str(epoch_count), '--schedule-epochs', str(spec.get('schedule_epochs', spec['epochs'])),
            '--samples', str(spec['samples']), '--batch', '2', '--workers', '2', '--lr', str(spec['lr']),
            '--rare-fraction', str(spec['rare']), '--loss', spec['loss'], '--seed', '42',
            '--max-seconds', str(min(spec['seconds'], max(120, remaining-120)))]
    if train_all:
        args.append('--train-all')
    if spec.get('legacy'):
        args.append('--legacy-amp')
    if (folder/'last.pt').exists():
        args.append('--resume')
    print(json.dumps({'stage': 'training_queued', 'id': name}), flush=True)
    command(name, args, note, phase_end)
    summary = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    result = {'id': name, 'spec': spec, 'path': str(folder), 'summary': summary}
    print(json.dumps({'stage': 'training_done', 'id': name, 'epochs': summary['epochs_completed'],
                      'score': summary['best_score'], 'seconds': summary['elapsed_seconds'],
                      'peak_gpu_bytes': summary['peak_gpu_bytes']}), flush=True)
    return result


def selected_row(run):
    rows = [json.loads(line) for line in (Path(run['path'])/'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    rows = [r for r in rows if r.get('dev') is not None]
    return max(rows, key=lambda x: x['dev']['miou'])


def record_failure(label, exc):
    result = {'id': label, 'error': type(exc).__name__ + ': ' + str(exc)}
    FAILURES.append(result)
    print(json.dumps({'stage': 'candidate_failed', **result}), flush=True)


def multi_train(jobs, note, phase_end, concurrency=1, epochs=None):
    results = {}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(train_case, spec, fold, note, phase_end, False, epochs): (spec, fold) for spec, fold in jobs}
        for future in as_completed(futures):
            spec, fold = futures[future]
            key = (spec['id'], fold)
            try:
                results[key] = future.result()
            except Exception as exc:
                record_failure('%s_fold%d' % (spec['id'], fold), exc)
    return results


def compare_fold(fold, course, base, tile, expert, note, until):
    out = RUN / ('fold_%d_predictions.json' % fold)
    args = [str(CODE/'v2_compare.py'), '--fold', str(fold), '--base-model', base['spec']['model'],
            '--base-checkpoint', str(Path(base['path'])/'best.pt'), '--tile-model', tile['spec']['model'],
            '--tile-checkpoint', str(Path(tile['path'])/'best.pt'), '--course-checkpoint', str(Path(course['path'])/'best.pt'),
            '--out', str(out)]
    if expert is not None:
        args += ['--expert-checkpoint', str(Path(expert['path'])/'best.pt')]
    command('compare_fold%d' % fold, args, note, until)
    return json.loads(out.read_text(encoding='utf-8'))


def ious(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    denom = matrix.sum(0) + matrix.sum(1) - np.diag(matrix)
    return np.divide(np.diag(matrix), denom, out=np.full(8, np.nan), where=denom > 0)


def summarize(results):
    names = set.intersection(*(set(x['variants']) for x in results))
    mats = {}
    for name in names:
        seq = {}
        for result in results:
            seq.update(result['variants'][name]['sequence_confusions'])
        if len(seq) != 20:
            raise ValueError('out-of-fold predictions do not cover all 20 sequences')
        mats[name] = {key: np.asarray(value, dtype=np.int64) for key, value in seq.items()}
    base = sum(mats['course_baseline'].values())
    base_iou = ious(base)
    mix_iou = ious(sum(mats['mix'].values()))
    table = {}
    keys = sorted(mats['course_baseline'])
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(keys), size=(1000, len(keys)))
    for name in sorted(names):
        total = sum(mats[name].values())
        iv = ious(total)
        deltas = []
        for draw in draws:
            a = sum(mats[name][keys[i]] for i in draw)
            b = sum(mats['course_baseline'][keys[i]] for i in draw)
            deltas.append(float(np.nanmean(ious(a))-np.nanmean(ious(b))))
        eligible = name == 'course_baseline' or (np.nanmean(iv) >= np.nanmean(base_iou) and iv[6] >= base_iou[6])
        if name in GATES:
            eligible = bool(eligible and np.nanmean(iv) > np.nanmean(mix_iou) and iv[6] > mix_iou[6])
        table[name] = {'miou': float(np.nanmean(iv)), 'human_iou': float(iv[6]),
                       'human_precision': float(total[6, 6] / max(1, total[:, 6].sum())),
                       'human_recall': float(total[6, 6] / max(1, total[6, :].sum())),
                       'class_iou': [float(v) if np.isfinite(v) else None for v in iv],
                       'class_delta_vs_course': [float(v) if np.isfinite(v) else None for v in iv-base_iou],
                       'confusion': total.tolist(), 'eligible': bool(eligible),
                       'delta_miou_vs_course': float(np.nanmean(iv)-np.nanmean(base_iou)),
                       'delta_miou_sequence_bootstrap_95': [float(x) for x in np.quantile(deltas, [.025, .975])]}
    best = max((n for n in table if table[n]['eligible']), key=lambda n: table[n]['miou'])
    return {'variants': table, 'selected_variant': best, 'folds': 3, 'sequences': 20,
            'selection_caveat': 'out-of-fold training predictions are also used for model and gate selection'}


def monitor(stop):
    while not stop.wait(10):
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu,temperature.gpu',
                                     '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=8, check=True)
            memory, utilization, temperature = [int(x.strip()) for x in result.stdout.splitlines()[0].split(',')]
            with (RUN/'resources.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps({'time': time.time(), 'gpu_memory_mib': memory,
                                    'gpu_utilization_pct': utilization, 'gpu_temperature_c': temperature})+'\n')
        except Exception:
            pass


def execute(note):
    if not INDEX.exists():
        raise RuntimeError('native-resolution cache is not ready')
    stage('legacy_expert_diagnosis')
    try:
        command('legacy_expert_diagnosis', [str(CODE/'v2_legacy_diagnosis.py'), '--out', str(RUN/'legacy_expert_diagnosis.json')],
                note, START+3600)
    except Exception as exc:
        record_failure('legacy_expert_diagnosis', exc)
    stage('paired_global_baselines')
    legacy = dict(COURSE, id='course_legacy_fp16_no_scaler', epochs=4, schedule_epochs=20, legacy=True, seconds=1200)
    jobs = [(COURSE, 0), (FPN, 0), (COURSE, 1), (COURSE, 2), (legacy, 0)]
    trained = multi_train(jobs, note, START+7*3600, concurrency=2)
    course = [trained.get((COURSE['id'], i)) for i in range(3)]
    if not all(course):
        raise RuntimeError('three corrected course baselines are required')
    base_spec, base = COURSE, course
    fpn0 = trained.get((FPN['id'], 0))
    if fpn0:
        f, c = selected_row(fpn0)['dev'], selected_row(course[0])['dev']
        if f['miou'] > c['miou'] + .002 and (f['iou']['human'] or 0) >= (c['iou']['human'] or 0)-.02:
            more = multi_train([(FPN, 1), (FPN, 2)], note, START+7*3600, concurrency=2)
            if all((FPN['id'], i) in more for i in (1, 2)):
                base_spec, base = FPN, [fpn0, more[(FPN['id'], 1)], more[(FPN['id'], 2)]]
    stage('native_resolution_pilots', global_model=base_spec['id'])
    # Compare the same two independent jobs for one epoch each; the second
    # measurement resumes their checkpoints, including RNG and scheduler state.
    pair = [(PILOTS[0], 0), (PILOTS[-1], 0)]
    started = time.monotonic()
    parallel = multi_train(pair, note, START+15*3600, concurrency=2, epochs=1)
    parallel_seconds = time.monotonic()-started
    started = time.monotonic()
    serial = multi_train([job for job in pair if (job[0]['id'], job[1]) in parallel],
                         note, START+15*3600, concurrency=1, epochs=2)
    serial_seconds = time.monotonic()-started
    concurrency = 2 if len(parallel) == len(serial) == 2 and parallel_seconds*1.05 < serial_seconds else 1
    atomic_json(RUN/'efficiency.json', {'parallel_two_first_epochs_seconds': round(parallel_seconds, 2),
                'serial_two_second_epochs_seconds': round(serial_seconds, 2), 'tile_concurrency': concurrency,
                'caveat': 'different training epochs; exploratory scheduling comparison'})
    stage('tile_efficiency_measured', concurrency=concurrency, parallel_seconds=round(parallel_seconds, 2),
          serial_seconds=round(serial_seconds, 2))
    failed_ids = {item['id'] for item in FAILURES}
    pilot_jobs = [(x, 0) for x in PILOTS if x['id']+'_fold0' not in failed_ids]
    pilot_runs = multi_train(pilot_jobs, note, START+15*3600, concurrency=concurrency)
    if not pilot_runs:
        raise RuntimeError('no native-resolution candidate completed')
    def pilot_rank(run):
        d = selected_row(run)['dev']
        return d['miou'] + .25 * (d['iou']['human'] or 0.)
    tile0 = max(pilot_runs.values(), key=pilot_rank)
    tile_spec = tile0['spec']
    remaining = multi_train([(tile_spec, 1), (tile_spec, 2)], note, START+15*3600, concurrency=concurrency)
    if not all((tile_spec['id'], i) in remaining for i in (1, 2)):
        raise RuntimeError('three native-resolution folds are required')
    tile = [tile0, remaining[(tile_spec['id'], 1)], remaining[(tile_spec['id'], 2)]]
    stage('human_expert_three_folds', tile_model=tile_spec['id'])
    expert_runs = multi_train([(EXPERT, i) for i in range(3)], note, START+19*3600, concurrency=1)
    expert_ok = all((EXPERT['id'], i) in expert_runs for i in range(3))
    if not expert_ok:
        print('expert_gate_disabled_incomplete_three_fold_evidence', flush=True)
    stage('paired_out_of_fold_comparison', expert_available=expert_ok)
    folds = []
    for i in range(3):
        folds.append(compare_fold(i, course[i], base[i], tile[i], expert_runs.get((EXPERT['id'], i)) if expert_ok else None,
                                  note, START+19*3600))
    comparison = summarize(folds)
    atomic_json(RUN/'cross_validation.json', comparison)
    notify(note, 'metric', cv_miou=comparison['variants'][comparison['selected_variant']]['miou'])
    variant = comparison['selected_variant']
    chosen = []
    if variant == 'course_baseline':
        chosen = [('global', COURSE, course)]
    elif variant == 'global':
        chosen = [('global', base_spec, base)]
    elif variant == 'tile':
        chosen = [('tile', tile_spec, tile)]
    else:
        chosen = [('global', base_spec, base), ('tile', tile_spec, tile)]
    frozen = {'variant': variant, 'global_candidate': base_spec, 'tile_candidate': tile_spec,
              'fold_scores': comparison, 'decision_before_official_validation': True,
              'official_validation_history': 'previously observed; reference only, not blind'}
    atomic_json(RUN/'selection_before_official.json', frozen)
    stage('refit_on_all_training_sequences', selected_variant=variant)
    final_models = {}
    for kind, spec, runs in chosen:
        epochs = int(np.median([selected_row(r)['epoch'] for r in runs]))
        result = train_case(spec, 0, note, START+BUDGET-2*3600, train_all=True, epochs=epochs, label='final_'+kind)
        final_models[kind] = {'model': spec['model'], 'checkpoint': str(Path(result['path'])/'best.pt'),
                              'epochs': result['summary']['epochs_completed']}
    expert_checkpoint = None
    if variant in GATES:
        runs = [expert_runs[(EXPERT['id'], i)] for i in range(3)]
        epochs = int(np.median([selected_row(r)['epoch'] for r in runs]))
        result = train_case(EXPERT, 0, note, START+BUDGET-2*3600, train_all=True, epochs=epochs, label='final_expert')
        expert_checkpoint = str(Path(result['path'])/'best.pt')
    final = {'variant': variant, 'models': final_models, 'expert_checkpoint': expert_checkpoint, 'gate': GATES.get(variant)}
    atomic_json(RUN/'final_models.json', final)
    stage('official_reference_and_test_export')
    command('final_prediction', [str(CODE/'v2_predict.py'), '--selection', str(RUN/'final_models.json'),
            '--out', str(RUN), '--deadline', str(ABS_DEADLINE-10)], note)
    official = json.loads((RUN/'official_validation.json').read_text(encoding='utf-8'))
    predictions = json.loads((RUN/'prediction_summary.json').read_text(encoding='utf-8'))
    report = {'status': 'partial' if FAILURES else 'complete', 'cv': comparison, 'final': final,
              'official_validation': official, 'test_predictions': predictions, 'failures': FAILURES,
              'budget_seconds': BUDGET, 'elapsed_seconds': round(time.monotonic()-START),
              'original_reference_miou': .6022826895, 'official_history': 'not pristine blind'}
    atomic_json(RUN/'report.json', report)
    results = Path(os.environ['YUN_RESULTS_DIR'])
    for filename in ('report.json', 'selection_before_official.json', 'cross_validation.json', 'official_validation.json'):
        shutil.copy2(RUN/filename, results/filename)
    notify(note, 'metric', official_val_miou=official['metrics']['miou'])
    stage('complete', status=report['status'], official_miou=official['metrics']['miou'], test_images=predictions['test_images'])
    if FAILURES:
        raise RuntimeError('one or more stages failed; see report and logs')


def main():
    RUN.mkdir(parents=True, exist_ok=False)
    atomic_json(RUN/'config.json', {'course': COURSE, 'fpn': FPN, 'pilots': PILOTS, 'expert': EXPERT,
                'budget_seconds': BUDGET, 'seed': 42, 'fold_source': str(INDEX)})
    note = None
    try:
        email = json.loads((PRIVATE/'preferences.json').read_text(encoding='utf-8'))['email']
        note = Client(direct=True).experiment('空天智能语义分割 UAVid 原分辨率误差攻击 第2轮',
                                              heartbeat_timeout=300, heartbeat_interval=60,
                                              group='空天智能语义分割', email=email)
        note.__enter__()
        task_id = note.task['id']
        descriptor = os.open(PRIVATE/('task-'+task_id+'.json'), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'w', encoding='utf-8') as f:
            json.dump({'id': task_id, 'task_key': note.task['task_key']}, f)
        print('endnote_task_id='+task_id, flush=True)
        atomic_json(RUN/'notification_status.json', {'configured': True, 'task_id': task_id})
        notify(note, 'queued')
    except Exception as exc:
        if note is not None and note.task is not None:
            try:
                note.__exit__(type(exc), exc, exc.__traceback__)
            except Exception:
                pass
        note = None
        print('endnote_setup_error='+type(exc).__name__, flush=True)
        atomic_json(RUN/'notification_status.json', {'configured': False, 'error': type(exc).__name__})
    stop = threading.Event()
    thread = threading.Thread(target=monitor, args=(stop,), daemon=True)
    thread.start()
    try:
        execute(note)
    except BaseException as exc:
        atomic_json(RUN/'failures.json', FAILURES)
        stage('failed', error=type(exc).__name__, failure_count=len(FAILURES))
        close_note(note, type(exc), exc, exc.__traceback__)
        raise
    else:
        close_note(note)
    finally:
        stop.set()
        thread.join(15)


if __name__ == '__main__':
    main()
