"""Train a single UAVid baseline with a sequence-held-out development set."""
from pathlib import Path
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from data import UAVidImages, CLASS_NAMES
from models import create_model, forward
from expert_data import ExpertPatches


def save_atomic(obj, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def confusion_add(confusion, truth, pred):
    truth = truth.reshape(-1).long()
    pred = pred.reshape(-1).long()
    valid = (truth >= 0) & (truth < 8)
    return confusion + torch.bincount(truth[valid] * 8 + pred[valid], minlength=64).reshape(8, 8)


def score_confusion(confusion):
    matrix = confusion.detach().cpu().numpy().astype(np.float64)
    union = matrix.sum(1) + matrix.sum(0) - np.diag(matrix)
    iou = np.divide(np.diag(matrix), union, out=np.full(8, np.nan), where=union > 0)
    precision = np.divide(np.diag(matrix), matrix.sum(0), out=np.full(8, np.nan), where=matrix.sum(0) > 0)
    recall = np.divide(np.diag(matrix), matrix.sum(1), out=np.full(8, np.nan), where=matrix.sum(1) > 0)
    values = lambda a: {n: float(v) if np.isfinite(v) else None for n, v in zip(CLASS_NAMES, a)}
    return {'miou': float(np.nanmean(iou)), 'iou': {n: float(v) if np.isfinite(v) else None for n,v in zip(CLASS_NAMES, iou)},
            'precision': values(precision), 'recall': values(recall),
            'support_pixels': {n: int(v) for n, v in zip(CLASS_NAMES, matrix.sum(1))},
            'pixel_accuracy': float(np.diag(matrix).sum() / max(matrix.sum(), 1)), 'confusion': matrix.astype('int64').tolist()}


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype):
    model.eval()
    confusion = torch.zeros((8,8), dtype=torch.int64, device=device)
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(dtype=amp_dtype):
            logits, _ = forward(model, x)
        logits = F.interpolate(logits.float(), size=y.shape[-2:], mode='bilinear', align_corners=False)
        confusion = confusion_add(confusion, y, logits.argmax(1))
    return score_confusion(confusion)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--lr', type=float, default=0.0003)
    p.add_argument('--optimizer', choices=('adamw','sgd'), default='adamw')
    p.add_argument('--epochs', type=int, required=True)
    p.add_argument('--schedule-epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--train-all', action='store_true')
    p.add_argument('--limit-train', type=int)
    p.add_argument('--limit-dev', type=int)
    args = p.parse_args()
    out = Path(args.run_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not args.resume and (out/'last.pt').exists():
        raise RuntimeError('run already exists; explicit resume is required')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if not torch.cuda.is_available():
        raise RuntimeError('authorized CUDA device is unavailable')
    device = torch.device('cuda:0')
    is_expert = args.model == 'expert_mobile_unet'
    train_group = 'train_all' if args.train_all else 'train'
    train_set = ExpertPatches(train_group, augment=True) if is_expert else UAVidImages(train_group, augment=True)
    dev_set = None if args.train_all else (ExpertPatches('dev') if is_expert else UAVidImages('dev', full_labels=True))
    if args.limit_train is not None:
        train_set = Subset(train_set, range(min(args.limit_train, len(train_set))))
    if args.limit_dev is not None and dev_set is not None:
        dev_set = Subset(dev_set, range(min(args.limit_dev, len(dev_set))))
    if args.smoke:
        train_set = Subset(train_set, range(min(4, len(train_set))))
        if dev_set is not None:
            dev_set = Subset(dev_set, range(2))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=True, drop_last=True)
    dev_loader = None if dev_set is None else DataLoader(dev_set, batch_size=1, shuffle=False, num_workers=min(args.workers,2), pin_memory=True)
    model, coverage = create_model(args.model, pretrained=not args.smoke)
    model.to(device)
    class_weights = torch.tensor((1.,1.,1.,2.,1.,1.,4.,2.), device=device) if is_expert else None
    if args.optimizer == 'sgd':
        if args.model not in ('course_mobile','course_resnet50'):
            raise ValueError('course SGD groups require a course model')
        for layer in model.backbone.modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.momentum = 0.01
        optimizer = torch.optim.SGD([{'params':model.backbone.parameters(),'lr':0.1*args.lr},
                                     {'params':model.classifier.parameters(),'lr':args.lr}],
                                    lr=args.lr, momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if args.optimizer == 'sgd':
        total_steps = args.schedule_epochs * len(train_loader)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: (1-min(step/total_steps,1))**0.9)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.schedule_epochs)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=amp_dtype == torch.float16, init_scale=1024.0)
    start_epoch, best = 0, -1.0
    if args.resume:
        ckpt = torch.load(out/'last.pt', map_location='cpu', weights_only=True)
        if (ckpt['model_name'] != args.model or ckpt['lr'] != args.lr or ckpt['optimizer_name'] != args.optimizer
                or ckpt['batch_size'] != args.batch_size or ckpt['schedule_epochs'] != args.schedule_epochs
                or ckpt['train_all'] != args.train_all):
            raise ValueError('resume configuration mismatch')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch, best = int(ckpt['epoch']), float(ckpt['best'])
    print(json.dumps({'model':args.model,'params':sum(x.numel() for x in model.parameters()), 'pretrained_coverage':coverage,
                      'device':torch.cuda.get_device_name(0),'train':len(train_set),'dev':len(dev_set) if dev_set is not None else 0,'batch_size':args.batch_size,'train_all':args.train_all}),flush=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss, steps = 0.0, 0
        epoch_start = time.monotonic()
        for x, y, _ in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                logits, aux = forward(model, x)
                loss = F.cross_entropy(logits, y, weight=class_weights, ignore_index=255)
                if aux is not None:
                    loss = loss + 0.4 * F.cross_entropy(aux, y, weight=class_weights, ignore_index=255)
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite training loss')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if args.optimizer == 'sgd':
                scheduler.step()
            total_loss += float(loss.detach())
            steps += 1
            if args.smoke and steps >= 2:
                break
        if args.optimizer != 'sgd':
            scheduler.step()
        metrics = evaluate(model, dev_loader, device, amp_dtype) if dev_loader is not None else None
        row = {'model':args.model,'epoch':epoch+1,'train_loss':total_loss/max(1,steps),'dev':metrics,
               'lr':optimizer.param_groups[0]['lr'],'epoch_seconds':round(time.monotonic()-epoch_start,2),
               'peak_gpu_bytes':torch.cuda.max_memory_allocated()}
        with (out/'metrics.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, allow_nan=False)+'\n')
        print(json.dumps(row, allow_nan=False), flush=True)
        if not args.smoke:
            if args.train_all or metrics['miou'] > best:
                best = metrics['miou'] if metrics is not None else -1.0
                save_atomic({'model_name':args.model,'model':model.state_dict(),'epoch':epoch+1,'miou':best}, out/'best.pt')
            save_atomic({'model_name':args.model,'model':model.state_dict(),'epoch':epoch+1,'best':best,
                         'lr':args.lr,'optimizer_name':args.optimizer,'batch_size':args.batch_size,
                         'schedule_epochs':args.schedule_epochs,'train_all':args.train_all,'optimizer':optimizer.state_dict(),
                         'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict()}, out/'last.pt')
        else:
            best = metrics['miou'] if metrics is not None else -1.0
            break
    (out/'summary.json').write_text(json.dumps({'model':args.model,'epochs':epoch+1,'best_dev_miou':None if args.train_all else best,'train_all':args.train_all,
         'elapsed_seconds':round(time.monotonic()-started,2),'peak_gpu_bytes':torch.cuda.max_memory_allocated(),
         'smoke':args.smoke,'pretrained_coverage':coverage}, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
