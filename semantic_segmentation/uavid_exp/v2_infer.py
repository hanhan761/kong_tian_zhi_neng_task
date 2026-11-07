"""Full-resolution probability inference; no ground truth is used for gating."""
from pathlib import Path
import math

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from data import CACHE
from models import create_model, forward
from v2_data import ROOT, tensor_image

DEVICE = torch.device('cuda:0')
AMP = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def positions(length, size, stride):
    if length < size:
        raise ValueError('image is smaller than inference tile')
    result = list(range(0, length - size + 1, stride))
    if result[-1] != length - size:
        result.append(length - size)
    return result


def load_model(name, path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['model_name'] != name:
        raise ValueError('model/checkpoint mismatch')
    model, _ = create_model(name, pretrained=False)
    model.load_state_dict(saved['model'])
    return model.to(DEVICE).eval()


def image_for(record):
    cached = ROOT / 'images' / (record['key'] + '.npy')
    if cached.exists():
        return np.load(cached, mmap_mode='r', allow_pickle=False)
    with Image.open(record['image']) as im:
        return np.asarray(im.convert('RGB'))


def mask_for(record):
    cached = ROOT / 'masks' / (record['key'] + '.npy')
    if cached.exists():
        return np.load(cached, mmap_mode='r', allow_pickle=False)
    with Image.open(record['mask']) as im:
        return np.asarray(im).copy()


@torch.inference_mode()
def probabilities(model, image, mode, expert=False):
    h, w = image.shape[:2]
    if mode == 'global':
        small = Image.fromarray(np.asarray(image)).resize((1024, 512), Image.Resampling.BILINEAR)
        x = tensor_image(small).unsqueeze(0).to(DEVICE)
        with torch.cuda.amp.autocast(dtype=AMP):
            logits, _ = forward(model, x)
        return F.interpolate(logits.float().softmax(1), size=(h, w), mode='bilinear', align_corners=False)[0]
    size = 512 if expert else 1024
    stride = (size * 3 // 4) if expert else (size // 2)
    classes = 2 if expert else 8
    axis = torch.hann_window(size, periodic=False, device=DEVICE).clamp_min_(0.05)
    window = (axis[:, None] * axis[None, :]).clamp_min_(0.01)
    total = torch.zeros((classes, h, w), dtype=torch.float32, device=DEVICE)
    weights = torch.zeros((h, w), dtype=torch.float32, device=DEVICE)
    for y in positions(h, size, stride):
        for x in positions(w, size, stride):
            tile = tensor_image(image[y:y+size, x:x+size]).unsqueeze(0).to(DEVICE)
            with torch.cuda.amp.autocast(dtype=AMP):
                logits, _ = forward(model, tile)
            prob = logits.float().softmax(1)[0]
            total[:, y:y+size, x:x+size] += prob * window
            weights[y:y+size, x:x+size] += window
    if float(weights.min()) <= 0:
        raise RuntimeError('uncovered inference pixels')
    return total / weights.unsqueeze(0)


def gate_prediction(teacher, expert_human, gate):
    pred = teacher.argmax(0)
    if gate is None:
        return pred
    replace = (expert_human >= gate[0]) & (teacher[6] >= gate[1])
    return torch.where(replace, torch.full_like(pred, 6), pred)
