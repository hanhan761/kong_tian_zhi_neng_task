"""Choose a probability ensemble and rare-class expert on development data only."""
from pathlib import Path
import json
import math
import os
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import DATA, CACHE, MEAN, STD, PALETTE, UAVidImages, load_records, atomic_save
from models import create_model, forward
from train import confusion_add, score_confusion

DEVICE = torch.device('cuda:0')
AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
TARGET_CLASSES = (3, 6, 7)


def load_model(model_name, checkpoint):
    model, _ = create_model(model_name, pretrained=False)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved['model_name'] != model_name:
        raise ValueError('checkpoint model mismatch')
    model.load_state_dict(saved['model'])
    return model.to(DEVICE).eval()


@torch.no_grad()
def cache_probabilities(candidate, group, result_dir):
    out = Path(result_dir)/'probabilities'/candidate['id']/group
    out.mkdir(parents=True, exist_ok=True)
    ds = UAVidImages(group)
    model = load_model(candidate['model'], Path(candidate['path'])/'best.pt')
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=2, pin_memory=True)
    for x, _, keys in loader:
        with torch.cuda.amp.autocast(dtype=AMP_DTYPE):
            logits, _ = forward(model, x.to(DEVICE, non_blocking=True))
        probabilities = logits.float().softmax(1).cpu().numpy().astype(np.float16)
        for key, prob in zip(keys, probabilities):
            atomic_save(out/(key+'.npy'), prob)
    del model
    torch.cuda.empty_cache()
    print(json.dumps({'stage':'cached_predictions','candidate':candidate['id'],'group':group,'images':len(ds)}), flush=True)


def target_tensor(record):
    with Image.open(record['mask']) as im:
        target = np.asarray(im).copy()
    return torch.from_numpy(target.astype(np.int64)).to(DEVICE)


def load_probs(result_dir, candidate, group, key):
    file = Path(result_dir)/'probabilities'/candidate['id']/group/(key+'.npy')
    return torch.from_numpy(np.load(file, allow_pickle=False).astype(np.float32)).to(DEVICE)


def full_probs(probs, size):
    return F.interpolate(probs.unsqueeze(0), size=size, mode='bilinear', align_corners=False).squeeze(0)


@torch.no_grad()
def select_ensemble(ranked, result_dir):
    ranked = ranked[:3]
    records = load_records('dev')
    matrices = [torch.zeros((8,8),dtype=torch.int64,device=DEVICE) for _ in ranked]
    for i, record in enumerate(records, 1):
        target = target_tensor(record)
        total = None
        for j, candidate in enumerate(ranked):
            p = load_probs(result_dir, candidate, 'dev', record['key'])
            total = p if total is None else total+p
            prediction = full_probs(total/(j+1), target.shape).argmax(0)
            matrices[j] = confusion_add(matrices[j], target, prediction)
        if i%10==0:
            print(json.dumps({'stage':'ensemble_dev','images':i,'total':len(records)}),flush=True)
    scores = [score_confusion(m) for m in matrices]
    best = max(range(len(scores)), key=lambda j:scores[j]['miou'])
    return ranked[:best+1], [{'models':[x['id'] for x in ranked[:i+1]],'metrics':score} for i,score in enumerate(scores)]


def normalize_tile(tile):
    x = torch.from_numpy(np.array(tile,dtype=np.uint8)).permute(2,0,1).float().div_(255)
    return (x-MEAN)/STD


@torch.no_grad()
def expert_maps(model, image):
    height, width = image.shape[:2]
    side = 512
    ph, pw = math.ceil(height/side)*side, math.ceil(width/side)*side
    padded = np.pad(image, ((0,ph-height),(0,pw-width),(0,0)), mode='edge')
    labels = np.empty((ph,pw), dtype=np.uint8)
    confidence = np.empty((ph,pw), dtype=np.float16)
    positions = [(y,x) for y in range(0,ph,side) for x in range(0,pw,side)]
    for start in range(0,len(positions),4):
        sub = positions[start:start+4]
        batch = torch.stack([normalize_tile(padded[y:y+side,x:x+side]) for y,x in sub]).to(DEVICE)
        with torch.cuda.amp.autocast(dtype=AMP_DTYPE):
            logits, _ = forward(model,batch)
        values, ids = logits.float().softmax(1).max(1)
        values, ids = values.cpu().numpy().astype(np.float16), ids.cpu().numpy().astype(np.uint8)
        for k,(y,x) in enumerate(sub):
            labels[y:y+side,x:x+side] = ids[k]
            confidence[y:y+side,x:x+side] = values[k]
    return labels[:height,:width], confidence[:height,:width]


def gated_prediction(teacher, expert_ids, expert_conf, threshold):
    base_conf, base_ids = teacher.max(0)
    if threshold is None:
        return base_ids
    allowed = (expert_ids == TARGET_CLASSES[0]) | (expert_ids == TARGET_CLASSES[1]) | (expert_ids == TARGET_CLASSES[2])
    return torch.where(allowed & (expert_conf >= threshold) & (base_conf < 0.95), expert_ids.long(), base_ids)


@torch.no_grad()
def choose_expert(selected, expert, result_dir):
    model = load_model('expert_mobile_unet', Path(expert['path'])/'best.pt')
    thresholds = [None, 0.5, 0.7, 0.85, 0.95]
    mats = [torch.zeros((8,8),dtype=torch.int64,device=DEVICE) for _ in thresholds]
    records = load_records('dev')
    for i,record in enumerate(records,1):
        target = target_tensor(record)
        teacher = sum(load_probs(result_dir,c,'dev',record['key']) for c in selected)/len(selected)
        teacher = full_probs(teacher,target.shape)
        with Image.open(record['image']) as im:
            image = np.asarray(im.convert('RGB'))
        pred, conf = expert_maps(model,image)
        pred = torch.from_numpy(pred.astype(np.int64)).to(DEVICE)
        conf = torch.from_numpy(conf.astype(np.float32)).to(DEVICE)
        for j,threshold in enumerate(thresholds):
            mats[j] = confusion_add(mats[j], target, gated_prediction(teacher,pred,conf,threshold))
        if i%5==0:
            print(json.dumps({'stage':'expert_gate_dev','images':i,'total':len(records)}),flush=True)
    scores = [score_confusion(m) for m in mats]
    best = max(range(len(scores)),key=lambda j:scores[j]['miou'])
    del model
    torch.cuda.empty_cache()
    return thresholds[best], [{'threshold':t,'metrics':s} for t,s in zip(thresholds,scores)]


@torch.no_grad()
def evaluate_official(candidates, selected, expert, threshold, result_dir):
    records = load_records('official_val')
    keys = [c['id'] for c in candidates] + ['selected_final']
    mats = {k:torch.zeros((8,8),dtype=torch.int64,device=DEVICE) for k in keys}
    expert_model = load_model('expert_mobile_unet',Path(expert['path'])/'best.pt') if threshold is not None else None
    for i,record in enumerate(records,1):
        target = target_tensor(record)
        probs = {c['id']:load_probs(result_dir,c,'official_val',record['key']) for c in candidates}
        for candidate in candidates:
            prediction = full_probs(probs[candidate['id']],target.shape).argmax(0)
            mats[candidate['id']] = confusion_add(mats[candidate['id']],target,prediction)
        teacher = sum(probs[c['id']] for c in selected)/len(selected)
        teacher = full_probs(teacher,target.shape)
        if expert_model is not None:
            with Image.open(record['image']) as im:
                image = np.asarray(im.convert('RGB'))
            pred, conf = expert_maps(expert_model,image)
            pred = torch.from_numpy(pred.astype(np.int64)).to(DEVICE)
            conf = torch.from_numpy(conf.astype(np.float32)).to(DEVICE)
            final = gated_prediction(teacher,pred,conf,threshold)
        else:
            final = teacher.argmax(0)
        mats['selected_final'] = confusion_add(mats['selected_final'],target,final)
        if i%10==0:
            print(json.dumps({'stage':'official_validation','images':i,'total':len(records)}),flush=True)
    if expert_model is not None:
        del expert_model
    torch.cuda.empty_cache()
    return {k:score_confusion(v) for k,v in mats.items()}


@torch.no_grad()
def export_test_predictions(selected, expert, threshold, result_dir, limit=None):
    models = [load_model(c['model'],Path(c['path'])/'best.pt') for c in selected]
    expert_model = load_model('expert_mobile_unet',Path(expert['path'])/'best.pt') if threshold is not None else None
    test = sorted((DATA/'test').glob('seq*/Images/*.png'))
    if len(test) != 150:
        raise ValueError('unexpected test image count')
    if limit is not None:
        test = test[:limit]
    palette = np.asarray(PALETTE,dtype=np.uint8)
    root = Path(result_dir)/'test_predictions'
    for i,file in enumerate(test,1):
        with Image.open(file) as im:
            rgb = im.convert('RGB')
            image = np.asarray(rgb)
            small = rgb.resize((1024,512),Image.Resampling.BILINEAR)
        x = normalize_tile(small).unsqueeze(0).to(DEVICE)
        probabilities = []
        for model in models:
            with torch.cuda.amp.autocast(dtype=AMP_DTYPE):
                logits,_ = forward(model,x)
            probabilities.append(logits.float().softmax(1).squeeze(0))
        teacher = full_probs(sum(probabilities)/len(probabilities),image.shape[:2])
        if expert_model is not None:
            ids, conf = expert_maps(expert_model,image)
            ids = torch.from_numpy(ids.astype(np.int64)).to(DEVICE)
            conf = torch.from_numpy(conf.astype(np.float32)).to(DEVICE)
            pred = gated_prediction(teacher,ids,conf,threshold)
        else:
            pred = teacher.argmax(0)
        mask = pred.cpu().numpy().astype(np.uint8)
        seq = file.parent.parent.name
        raw = root/'trainids'/seq/file.name
        color = root/'color'/seq/file.name
        raw.parent.mkdir(parents=True,exist_ok=True)
        color.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(mask,mode='L').save(raw)
        Image.fromarray(palette[mask],mode='RGB').save(color)
        if i%10==0:
            print(json.dumps({'stage':'test_export','images':i,'total':len(test)}),flush=True)
    return {'images':len(test),'path':str(root)}
