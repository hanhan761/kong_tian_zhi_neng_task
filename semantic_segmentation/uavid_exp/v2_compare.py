"""Paired, full-resolution fold evaluation and optional expert gate probes."""
import argparse
import json

import numpy as np
import torch

from train import confusion_add, score_confusion
from v2_data import atomic_json, get_split
from v2_infer import DEVICE, gate_prediction, image_for, load_model, mask_for, probabilities

GATES = {'expert_080_005': (0.80, 0.05), 'expert_090_002': (0.90, 0.02),
         'expert_095_001': (0.95, 0.01)}


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fold', type=int, required=True)
    p.add_argument('--base-model', required=True)
    p.add_argument('--base-checkpoint', required=True)
    p.add_argument('--tile-model', required=True)
    p.add_argument('--tile-checkpoint', required=True)
    p.add_argument('--course-checkpoint')
    p.add_argument('--expert-checkpoint')
    p.add_argument('--out', required=True)
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    _, records = get_split(args.fold)
    if args.limit is not None:
        records = records[:args.limit]
    base = load_model(args.base_model, args.base_checkpoint)
    tile = load_model(args.tile_model, args.tile_checkpoint)
    same_course = args.base_model == 'course_resnet50' and args.base_checkpoint == args.course_checkpoint
    course = None if not args.course_checkpoint or same_course else load_model('course_resnet50', args.course_checkpoint)
    expert = load_model('expert_person', args.expert_checkpoint) if args.expert_checkpoint else None
    names = (['course_baseline'] if args.course_checkpoint else []) + ['global', 'tile', 'mix'] + (list(GATES) if expert else [])
    mats = {n: {} for n in names}
    diagnostic = {'human_gt': 0, 'expert_predicted_human': 0, 'expert_human_tp': 0}
    for i, r in enumerate(records, 1):
        image = image_for(r)
        truth = torch.from_numpy(np.array(mask_for(r), dtype=np.int64)).to(DEVICE)
        global_prob = probabilities(base, image, 'global')
        tile_prob = probabilities(tile, image, 'tile')
        mix = (global_prob + tile_prob) / 2
        predictions = {'global': global_prob.argmax(0), 'tile': tile_prob.argmax(0), 'mix': mix.argmax(0)}
        if args.course_checkpoint:
            predictions['course_baseline'] = predictions['global'] if same_course else probabilities(course, image, 'global').argmax(0)
        if expert:
            person = probabilities(expert, image, 'tile', expert=True)[1]
            for name, gate in GATES.items():
                predictions[name] = gate_prediction(mix, person, gate)
            actual = truth == 6
            diagnostic['human_gt'] += int(actual.sum())
            diagnostic['expert_predicted_human'] += int((person >= .5).sum())
            diagnostic['expert_human_tp'] += int(((person >= .5) & actual).sum())
        for name, pred in predictions.items():
            seq = r['seq']
            if seq not in mats[name]:
                mats[name][seq] = torch.zeros((8, 8), dtype=torch.int64, device=DEVICE)
            mats[name][seq] = confusion_add(mats[name][seq], truth, pred)
        if i % 10 == 0 or i == len(records):
            print(json.dumps({'stage': 'paired_fold_evaluation', 'fold': args.fold,
                              'images': i, 'total': len(records)}), flush=True)
    result = {'fold': args.fold, 'images': len(records), 'variants': {}, 'expert_diagnostic': diagnostic}
    for name in names:
        total = sum(mats[name].values(), torch.zeros((8, 8), dtype=torch.int64, device=DEVICE))
        result['variants'][name] = {'metrics': score_confusion(total),
                                    'sequence_confusions': {k: v.cpu().tolist() for k, v in mats[name].items()}}
    atomic_json(args.out, result)
    print(json.dumps({'stage': 'fold_done', 'fold': args.fold,
                      'scores': {n: result['variants'][n]['metrics']['miou'] for n in names}}), flush=True)


if __name__ == '__main__':
    main()
