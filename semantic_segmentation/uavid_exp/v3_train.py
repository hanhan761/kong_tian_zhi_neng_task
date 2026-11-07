"""Apply nested sequence splits and optional expert hard negatives to v2 training."""
from pathlib import Path
import argparse
import json
import sys

import torch

import v2_train
from v2_data import get_split, atomic_json
from v3_data import HardNegativeImages, nested_split


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--v3-stage', choices=('inner', 'outer', 'final'), required=True)
    parser.add_argument('--v3-hard')
    parser.add_argument('--v3-init')
    extra, rest = parser.parse_known_args()
    basic = argparse.ArgumentParser(add_help=False)
    basic.add_argument('--out', required=True)
    basic.add_argument('--fold', type=int, default=0)
    basic.add_argument('--resume', action='store_true')
    known, _ = basic.parse_known_args(rest)
    if extra.v3_stage == 'inner':
        split = nested_split(known.fold)
        train, dev = split['inner_train'], split['inner_val']
    elif extra.v3_stage == 'outer':
        split = nested_split(known.fold)
        train, dev = split['outer_train'], []
    else:
        train, dev = get_split(known.fold, train_all=True)
    train_seq = sorted(set(r['seq'] for r in train))
    dev_seq = sorted(set(r['seq'] for r in dev))
    contract = {'stage': extra.v3_stage, 'fold': known.fold, 'train_sequences': train_seq,
                'dev_sequences': dev_seq, 'hard_negative_file': extra.v3_hard,
                'initial_checkpoint': extra.v3_init}
    out = Path(known.out)
    out.mkdir(parents=True, exist_ok=True)
    contract_path = out/'v3_contract.json'
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding='utf-8')) != contract:
            raise ValueError('nested split or initialization contract changed')
    else:
        atomic_json(contract_path, contract)
    v2_train.get_split = lambda fold, train_all=False: (train, dev)
    v2_train.TrainingImages = lambda records, mode, rare_fraction=0., expert=False, samples=300: HardNegativeImages(
        records, mode, rare_fraction, expert, samples, hard_path=extra.v3_hard)
    create = v2_train.create_model
    def create_model(name, pretrained=True):
        model, coverage = create(name, pretrained=pretrained)
        if extra.v3_init and not known.resume:
            ckpt = torch.load(extra.v3_init, map_location='cpu', weights_only=True)
            if ckpt['model_name'] != name:
                raise ValueError('initial checkpoint model mismatch')
            model.load_state_dict(ckpt['model'])
        return model, coverage
    v2_train.create_model = create_model
    if extra.v3_stage != 'inner' and '--train-all' not in rest:
        rest.append('--train-all')
    if extra.v3_stage == 'inner' and '--train-all' in rest:
        raise ValueError('inner training cannot use train-all')
    sys.argv = [sys.argv[0], *rest]
    print(json.dumps({'stage': 'nested_contract', **contract}), flush=True)
    v2_train.main()


if __name__ == '__main__':
    main()
