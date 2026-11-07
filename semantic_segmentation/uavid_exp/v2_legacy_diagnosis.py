"""Explain old expert gate coverage on the previously used development split."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
import torch

from data import load_records
from ensemble import load_model, expert_maps, load_probs, full_probs, target_tensor
from v2_data import atomic_json

RUN = Path('/data/aerospace-intelligence/runs/uavid-20261103-v1')
SELECTED = [{'id': 'course_resnet50_adamw_lrhalf'}, {'id': 'fpn_resnet34'}]


@torch.inference_mode()
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    expert = load_model('expert_mobile_unet', RUN / 'models/expert_mobile_unet/best.pt')
    records = load_records('dev')[:args.limit]
    sums = {key: 0 for key in ('pixels', 'human_gt', 'expert_human', 'expert_human_tp',
            'expert_human_05', 'expert_human_05_tp', 'gate05', 'gate05_human_tp',
            'human_base_high_conf', 'human_base_correct')}
    for i, r in enumerate(records, 1):
        y = target_tensor(r)
        teacher = sum(load_probs(RUN, c, 'dev', r['key']) for c in SELECTED) / len(SELECTED)
        teacher = full_probs(teacher, y.shape)
        base_conf, base_ids = teacher.max(0)
        with Image.open(r['image']) as im:
            image = np.asarray(im.convert('RGB'))
        ids, conf = expert_maps(expert, image)
        ids = torch.from_numpy(ids.astype(np.int64)).to(y.device)
        conf = torch.from_numpy(conf.astype(np.float32)).to(y.device)
        actual = y == 6
        pred = ids == 6
        gate = pred & (conf >= .5) & (base_conf < .95)
        values = {'pixels': y.numel(), 'human_gt': actual.sum(), 'expert_human': pred.sum(),
                  'expert_human_tp': (pred & actual).sum(), 'expert_human_05': (pred & (conf >= .5)).sum(),
                  'expert_human_05_tp': (pred & (conf >= .5) & actual).sum(), 'gate05': gate.sum(),
                  'gate05_human_tp': (gate & actual).sum(), 'human_base_high_conf': (actual & (base_conf >= .95)).sum(),
                  'human_base_correct': (actual & (base_ids == 6)).sum()}
        for key, value in values.items():
            sums[key] += int(value)
        if i % 10 == 0 or i == len(records):
            print(json.dumps({'stage': 'legacy_expert_diagnosis', 'images': i, 'total': len(records)}), flush=True)
    result = {'images': len(records), 'counts': sums, 'definition': 'pixel counts on old 40-image development split',
              'note': 'diagnostic labels are never used for inference gating'}
    atomic_json(args.out, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
