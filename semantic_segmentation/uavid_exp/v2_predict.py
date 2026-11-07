"""Evaluate frozen final models once and export unlabeled test predictions."""
from pathlib import Path
import argparse
import json
import time

import numpy as np
from PIL import Image
import torch

from data import DATA, PALETTE, load_records
from train import confusion_add, score_confusion
from v2_data import atomic_json, get_split
from v2_infer import DEVICE, gate_prediction, image_for, load_model, mask_for, probabilities


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--selection', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--deadline', type=float, required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    selection = json.loads(Path(args.selection).read_text(encoding='utf-8'))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    models = {key: load_model(spec['model'], spec['checkpoint']) for key, spec in selection['models'].items()}
    expert = load_model('expert_person', selection['expert_checkpoint']) if selection.get('expert_checkpoint') else None
    gate = selection.get('gate')
    def predict(image):
        parts = [probabilities(model, image, mode) for mode, model in models.items()]
        teacher = sum(parts) / len(parts)
        if expert is not None:
            person = probabilities(expert, image, 'tile', expert=True)[1]
            return gate_prediction(teacher, person, gate)
        return teacher.argmax(0)
    def ensure_time():
        if time.time() >= args.deadline:
            raise TimeoutError('24-hour experiment deadline reached')
    if args.smoke:
        _, records = get_split(0)
        image = image_for(records[0])
        pred = predict(image)
        if tuple(pred.shape) != tuple(image.shape[:2]) or int(pred.max()) > 7 or int(pred.min()) < 0:
            raise ValueError('invalid smoke prediction')
        atomic_json(out/'smoke_prediction.json', {'shape': list(pred.shape), 'input': 'training-fold development image'})
        print(json.dumps({'stage': 'prediction_smoke_passed', 'shape': list(pred.shape)}), flush=True)
        return
    val = load_records('official_val')
    if len(val) != 70:
        raise ValueError('unexpected official validation size')
    mat = torch.zeros((8, 8), dtype=torch.int64, device=DEVICE)
    for i, record in enumerate(val, 1):
        ensure_time()
        pred = predict(image_for(record))
        truth = torch.from_numpy(np.array(mask_for(record), dtype=np.int64)).to(DEVICE)
        mat = confusion_add(mat, truth, pred)
        if i % 10 == 0 or i == len(val):
            print(json.dumps({'stage': 'official_validation', 'images': i, 'total': len(val)}), flush=True)
    atomic_json(out / 'official_validation.json', {'images': len(val), 'metrics': score_confusion(mat),
                'qualification': 'historically exposed; final reference, not a pristine blind holdout'})
    test = sorted((DATA / 'test').glob('seq*/Images/*.png'))
    if len(test) != 150:
        raise ValueError('unexpected test image count')
    palette = np.asarray(PALETTE, dtype=np.uint8)
    for i, file in enumerate(test, 1):
        ensure_time()
        with Image.open(file) as im:
            image = np.asarray(im.convert('RGB'))
        mask = predict(image).cpu().numpy().astype(np.uint8)
        seq = file.parent.parent.name
        for kind, arr, mode in (('trainids', mask, 'L'), ('color', palette[mask], 'RGB')):
            dest = out / 'test_predictions' / kind / seq / file.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix('.png.tmp')
            Image.fromarray(arr, mode=mode).save(tmp, format='PNG')
            tmp.replace(dest)
        if i % 10 == 0 or i == len(test):
            print(json.dumps({'stage': 'test_export', 'images': i, 'total': len(test)}), flush=True)
    atomic_json(out / 'prediction_summary.json', {'test_images': len(test), 'labels_available': False,
                'path': str(out / 'test_predictions')})


if __name__ == '__main__':
    main()
