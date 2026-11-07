"""Frozen inference on previously observed validation and unlabeled test images."""
from pathlib import Path
import argparse, json, time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from data import DATA, PALETTE, load_records
from train import confusion_add, score_confusion
from v2_data import atomic_json
from v2_infer import DEVICE, image_for, mask_for, load_model, probabilities

@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--selection', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--deadline', type=float, required=True)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    selection = json.loads(Path(a.selection).read_text(encoding='utf-8'))
    models = {k: load_model(v['model'], v['checkpoint']) for k, v in selection['models'].items()}
    strategy = selection['strategy']
    out = Path(a.out)
    def infer(image):
        h, w = image.shape[:2]
        def prob(key, img):
            return probabilities(models[key], img, 'global' if key == 'global' else 'tile', expert=key.startswith('expert'))
        if strategy['type'] == 'mix':
            parts = {'global': None, 'tile': None}
            for k, v in strategy['weights'].items():
                if v:
                    parts[k] = prob(k, image)
            return sum(parts[k]*v for k, v in strategy['weights'].items() if v).argmax(0)
        base = (prob('global', image) + prob('tile', image)) / 2
        if strategy['type'] == 'expert':
            pred = base.argmax(0)
            q = prob(strategy['expert'], image)[1]
            gate = ((pred == 0) | (pred == 2)) & (q >= strategy['q_threshold']) & (base[6] >= strategy['person_min'])
            return torch.where(gate, torch.full_like(pred, 6), pred)
        if strategy['type'] == 'tta':
            if strategy['transform'] == 'flip':
                trans = np.ascontiguousarray(image[:, ::-1])
                pp = ((prob('global', trans) + prob('tile', trans))/2).flip(-1)
            else:
                size = (max(1024, int(w*.8)//32*32), max(1024, int(h*.8)//32*32))
                trans = np.asarray(Image.fromarray(np.asarray(image)).resize(size, Image.Resampling.BILINEAR))
                pp = (prob('global', trans) + prob('tile', trans))/2
                pp = F.interpolate(pp.unsqueeze(0), size=(h,w), mode='bilinear', align_corners=False)[0]
            return ((base+pp)/2).argmax(0)
        raise ValueError('unknown strategy')
    val = load_records('official_val')
    if len(val) != 70: raise ValueError('unexpected validation count')
    if a.smoke:
        pred = infer(image_for(val[0]))
        assert tuple(pred.shape) == tuple(image_for(val[0]).shape[:2])
        atomic_json(out/'smoke_prediction.json', {'shape': list(pred.shape)})
        return
    mat = torch.zeros((8,8), dtype=torch.int64, device=DEVICE)
    for i, record in enumerate(val, 1):
        if time.time() >= a.deadline: raise TimeoutError('deadline')
        pred = infer(image_for(record))
        truth = torch.from_numpy(np.array(mask_for(record), dtype=np.int64)).to(DEVICE)
        mat = confusion_add(mat, truth, pred)
        if i % 10 == 0: print(json.dumps({'stage':'official_validation','images':i}), flush=True)
    atomic_json(out/'official_validation.json', {'images':70, 'metrics':score_confusion(mat), 'qualification':'previously observed; reference only'})
    files = sorted((DATA/'test').glob('seq*/Images/*.png'))
    if len(files) != 150: raise ValueError('unexpected test count')
    palette = np.asarray(PALETTE, dtype=np.uint8)
    for i, file in enumerate(files,1):
        if time.time() >= a.deadline: raise TimeoutError('deadline')
        with Image.open(file) as im: image = np.asarray(im.convert('RGB'))
        mask = infer(image).cpu().numpy().astype(np.uint8)
        for kind, arr, mode in (('trainids',mask,'L'),('color',palette[mask],'RGB')):
            dest = out/'test_predictions'/kind/file.parent.parent.name/file.name
            dest.parent.mkdir(parents=True,exist_ok=True)
            tmp = dest.with_suffix('.png.tmp')
            Image.fromarray(arr,mode=mode).save(tmp,format='PNG')
            tmp.replace(dest)
        if i % 10 == 0: print(json.dumps({'stage':'test_export','images':i}), flush=True)
    atomic_json(out/'prediction_summary.json', {'test_images':150,'labels_available':False,'path':str(out/'test_predictions')})

if __name__ == '__main__': main()
