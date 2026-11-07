"""Mine false-positive human pixels only within the model's training sequences."""
from pathlib import Path
import argparse
import json

import numpy as np
import torch

from v2_data import atomic_json, get_split
from v2_infer import DEVICE, image_for, load_model, mask_for, probabilities
from v3_data import nested_split


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--stage', choices=('inner', 'outer', 'final'), required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--images', type=int, default=8)
    args = p.parse_args()
    contract = json.loads((Path(args.checkpoint).parent/'v3_contract.json').read_text(encoding='utf-8'))
    if contract['stage'] != args.stage or contract['fold'] != args.fold:
        raise ValueError('expert checkpoint belongs to another split')
    records = get_split(args.fold, train_all=True)[0] if args.stage == 'final' else nested_split(args.fold)[args.stage+'_train']
    if sorted({r['seq'] for r in records}) != contract['train_sequences']:
        raise ValueError('mining sequence mismatch')
    # One image per sequence first, then remaining images; stable and label-independent selection.
    selected, visited = [], set()
    for r in sorted(records, key=lambda x: (x['key'],)):
        if r['seq'] not in visited:
            selected.append(r)
            visited.add(r['seq'])
            if len(selected) >= args.images:
                break
    model = load_model('expert_person', args.checkpoint)
    rng = np.random.default_rng(42 + args.fold)
    entries = []
    for i, r in enumerate(selected, 1):
        image = image_for(r)
        truth = torch.from_numpy(np.array(mask_for(r), dtype=np.int64)).to(DEVICE)
        q = probabilities(model, image, 'tile', expert=True)[1]
        mask = ((q >= .4) & (truth != 6) & (truth < 8)).flatten()
        ids = torch.nonzero(mask, as_tuple=False).flatten().cpu().numpy()
        if len(ids) > 2048:
            ids = rng.choice(ids, 2048, replace=False)
        entries.append({'key': r['key'], 'coords': ids.astype(np.int64).tolist()})
        print(json.dumps({'stage': 'hard_negative_mining', 'images': i, 'total': len(selected),
                          'sampled_false_positive_pixels': len(ids)}), flush=True)
    atomic_json(args.out, {'fold': args.fold, 'stage': args.stage, 'threshold': .4,
                           'train_sequences': contract['train_sequences'], 'entries': entries})


if __name__ == '__main__':
    main()
