"""Nested held-out evaluation for probability fusion, TTA and human experts."""
from pathlib import Path
import argparse
import json

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from train import confusion_add, score_confusion
from v2_data import atomic_json
from v2_infer import DEVICE, image_for, load_model, mask_for, probabilities
from v3_data import nested_split


def inner_records(records):
    chosen = []
    for seq in sorted({r['seq'] for r in records}):
        group = sorted((r for r in records if r['seq'] == seq), key=lambda x: x['key'])
        dense = max(group, key=lambda x: (x['counts']['6'], x['key']))
        other = next((x for x in group if x['key'] != dense['key']), dense)
        chosen.extend((dense, other))
    return chosen


def specs(models):
    out = {'baseline': {'family': 'baseline', 'type': 'mix', 'weights': {'global': .5, 'tile': .5}}}
    for g in (0., .25, .75, 1.):
        out['weight_'+str(g)] = {'family': 'weights', 'type': 'mix', 'weights': {'global': g, 'tile': 1-g}}
    for name in ('sfa', 'conv'):
        if name in models:
            for weight in (.25, .5):
                out[name+'_'+str(weight)] = {'family': name, 'type': 'mix',
                    'weights': {'global': (1-weight)/2, 'tile': (1-weight)/2, name: weight}}
    out['tta_flip'] = {'family': 'tta', 'type': 'tta', 'transform': 'flip'}
    out['tta_scale08'] = {'family': 'tta', 'type': 'tta', 'transform': 'scale08'}
    for expert in ('expert_clean', 'expert_hard'):
        if expert in models:
            for q in (.3, .4, .5, .6):
                for person in (0., .01, .05):
                    name = '%s_q%s_p%s' % (expert, q, person)
                    out[name] = {'family': 'expert', 'type': 'expert', 'expert': expert,
                                 'q_threshold': q, 'person_min': person}
    return out


def probability(model, image, kind):
    return probabilities(model, image, 'global' if kind == 'global' else 'tile',
                         expert=kind.startswith('expert'))


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fold', type=int, choices=(0, 1, 2), required=True)
    p.add_argument('--stage', choices=('inner', 'outer'), required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    models = {k: load_model(v['model'], v['checkpoint']) for k, v in config['models'].items()}
    if not {'global', 'tile'}.issubset(models):
        raise ValueError('both baseline branches are required')
    split = nested_split(args.fold)
    records = inner_records(split['inner_val']) if args.stage == 'inner' else split['outer_val']
    if args.limit is not None:
        records = records[:args.limit]
    variants = specs(models) if args.stage == 'inner' else {'baseline': specs(models)['baseline'], **config['choices']}
    counts = {name: {} for name in variants}
    diagnostic = {'images': 0, 'human_pixels': 0, 'baseline_missed_human': 0}
    for i, rec in enumerate(records, 1):
        image = image_for(rec)
        h, w = image.shape[:2]
        y = torch.from_numpy(np.array(mask_for(rec), dtype=np.int64)).to(DEVICE)
        needed = {'global', 'tile'}
        for v in variants.values():
            needed.update(k for k, value in v.get('weights', {}).items() if value)
            if v['type'] == 'expert':
                needed.add(v['expert'])
        probs = {k: probability(models[k], image, k) for k in sorted(needed)}
        base = (probs['global'] + probs['tile']) / 2
        base_pred = base.argmax(0)
        diagnostic['images'] += 1
        diagnostic['human_pixels'] += int((y == 6).sum())
        diagnostic['baseline_missed_human'] += int(((y == 6) & (base_pred != 6)).sum())
        tta = {}
        for variant in variants.values():
            if variant['type'] != 'tta' or variant['transform'] in tta:
                continue
            if variant['transform'] == 'flip':
                transformed = np.ascontiguousarray(image[:, ::-1])
                pp = (probability(models['global'], transformed, 'global') +
                      probability(models['tile'], transformed, 'tile')) / 2
                tta['flip'] = (base + pp.flip(-1)) / 2
            elif variant['transform'] == 'scale08':
                wh = (max(1024, int(w*.8)//32*32), max(1024, int(h*.8)//32*32))
                transformed = np.asarray(Image.fromarray(np.asarray(image)).resize(wh, Image.Resampling.BILINEAR))
                pp = (probability(models['global'], transformed, 'global') +
                      probability(models['tile'], transformed, 'tile')) / 2
                pp = F.interpolate(pp.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False)[0]
                tta['scale08'] = (base + pp) / 2
        for name, variant in variants.items():
            if variant['type'] == 'mix':
                prediction = sum(probs[k]*v for k, v in variant['weights'].items() if v).argmax(0)
            elif variant['type'] == 'tta':
                prediction = tta[variant['transform']].argmax(0)
            elif variant['type'] == 'expert':
                q = probs[variant['expert']][1]
                allowed = (base_pred == 0) | (base_pred == 2)
                change = allowed & (q >= variant['q_threshold']) & (base[6] >= variant['person_min'])
                prediction = torch.where(change, torch.full_like(base_pred, 6), base_pred)
            else:
                raise ValueError('unknown inference strategy')
            seq = rec['seq']
            if seq not in counts[name]:
                counts[name][seq] = torch.zeros((8, 8), dtype=torch.int64, device=DEVICE)
            counts[name][seq] = confusion_add(counts[name][seq], y, prediction)
        if i % 5 == 0 or i == len(records):
            print(json.dumps({'stage': args.stage+'_evaluation', 'fold': args.fold,
                              'images': i, 'total': len(records)}), flush=True)
    result = {'fold': args.fold, 'stage': args.stage, 'images': len(records),
              'diagnostic': diagnostic, 'variants': {}}
    for name, variant in variants.items():
        matrices = counts[name]
        total = sum(matrices.values(), torch.zeros((8, 8), dtype=torch.int64, device=DEVICE))
        result['variants'][name] = {'config': variant, 'metrics': score_confusion(total),
                                    'sequence_confusions': {s: m.cpu().tolist() for s, m in matrices.items()}}
    if args.stage == 'inner':
        base = result['variants']['baseline']['metrics']
        choices = {}
        for family in ('weights', 'sfa', 'conv', 'tta', 'expert'):
            candidates = [(name, row) for name, row in result['variants'].items() if row['config']['family'] == family]
            candidates = [(name, row) for name, row in candidates
                          if (row['metrics']['iou']['human'] or 0.) >= (base['iou']['human'] or 0.)-.02]
            if candidates:
                name, row = max(candidates, key=lambda x: x[1]['metrics']['miou'])
                choices[family] = {'name': name, 'config': row['config'], 'inner_miou': row['metrics']['miou'],
                                   'inner_gain': row['metrics']['miou']-base['miou']}
        result['choices'] = choices
    atomic_json(args.out, result)
    print(json.dumps({'stage': args.stage+'_done', 'fold': args.fold, 'images': len(records),
                      'scores': {n: round(x['metrics']['miou'], 6) for n, x in result['variants'].items()}}), flush=True)


if __name__ == '__main__':
    main()
