"""One bounded, resumable UAVid training run with fixed sequence folds."""
from pathlib import Path
import argparse
import json
import random
import time

import numpy as np
from segmentation_models_pytorch.losses import LovaszLoss
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import create_model, forward
from train import confusion_add, save_atomic, score_confusion
from v2_data import TrainingImages, get_split, atomic_json
from v2_infer import DEVICE, image_for, mask_for, probabilities

LOVASZ = LovaszLoss(mode='multiclass', ignore_index=255, from_logits=True)


def rng_save():
    n = np.random.get_state()
    return {'python': random.getstate(), 'numpy': (n[0], n[1].tolist(), n[2], n[3], n[4]),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def rng_restore(rng):
    random.setstate(rng['python'])
    n = rng['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(rng['torch'])
    torch.cuda.set_rng_state_all(rng['cuda'])


def loss_for(logits, target, expert, loss_name):
    weights = torch.tensor([1., 12.], device=logits.device) if expert else None
    loss = F.cross_entropy(logits, target, weight=weights, ignore_index=255)
    if loss_name == 'ce_dice':
        probs = logits.float().softmax(1)
        valid = target != 255
        terms = []
        for cls in ((1,) if expert else (3, 6, 7)):
            actual = (target == cls) & valid
            if actual.any():
                pred = probs[:, cls] * valid
                terms.append(1 - (2 * (pred * actual).sum() + 1) / (pred.sum() + actual.sum() + 1))
        if terms:
            loss = loss + 0.3 * torch.stack(terms).mean()
    elif loss_name == 'ce_lovasz':
        if expert:
            raise ValueError('Lovasz pilot is configured for the eight-class model')
        loss = loss + 0.3 * LOVASZ(logits.float(), target)
    return loss


@torch.inference_mode()
def evaluate_subset(model, records, mode, expert, smoke=False):
    chosen, count = [], {}
    for r in sorted(records, key=lambda x: (x['seq'], x['key'])):
        if count.get(r['seq'], 0) < (1 if smoke else 2):
            chosen.append(r)
            count[r['seq']] = count.get(r['seq'], 0) + 1
        if smoke and chosen:
            break
    mat = torch.zeros((8, 8), dtype=torch.int64, device=DEVICE)
    binary = torch.zeros(3, dtype=torch.int64, device=DEVICE)
    model.eval()
    for r in chosen:
        prob = probabilities(model, image_for(r), mode, expert=expert)
        truth = torch.from_numpy(np.array(mask_for(r), dtype=np.int64)).to(DEVICE)
        if expert:
            pred = prob[1] >= 0.5
            actual, valid = truth == 6, truth != 255
            binary += torch.stack(((pred & actual & valid).sum(),
                                   (pred & ~actual & valid).sum(), (~pred & actual & valid).sum()))
        else:
            mat = confusion_add(mat, truth, prob.argmax(0))
    if expert:
        tp, fp, fn = [int(x) for x in binary.cpu()]
        return {'miou': tp / max(1, tp + fp + fn), 'human_tp': tp, 'human_fp': fp, 'human_fn': fn,
                'images': len(chosen), 'score_name': 'human_iou'}
    result = score_confusion(mat)
    result['images'] = len(chosen)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--fold', type=int, choices=(0, 1, 2), default=0)
    p.add_argument('--mode', choices=('global', 'tile'), required=True)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--schedule-epochs', type=int, default=12)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--samples', type=int, default=300)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--lr', type=float, default=0.00015)
    p.add_argument('--rare-fraction', type=float, default=0.)
    p.add_argument('--loss', choices=('ce', 'ce_dice', 'ce_lovasz'), default='ce')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-seconds', type=float, default=3600.)
    p.add_argument('--train-all', action='store_true')
    p.add_argument('--legacy-amp', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not args.resume and (out / 'last.pt').exists():
        raise RuntimeError('existing run requires explicit --resume')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA not available')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    train, dev = get_split(args.fold, args.train_all)
    expert = args.model == 'expert_person'
    dataset = TrainingImages(train, args.mode, args.rare_fraction, expert,
                             samples=4 if args.smoke else args.samples)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        pin_memory=True, drop_last=True)
    if len(loader) == 0:
        raise ValueError('empty training loader')
    model, coverage = create_model(args.model, pretrained=not args.smoke)
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.schedule_epochs)
    dtype = torch.float16 if args.legacy_amp else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16 and not args.legacy_amp), init_scale=1024.)
    config = {k: getattr(args, k) for k in ('model', 'fold', 'mode', 'schedule_epochs', 'batch', 'samples',
              'lr', 'rare_fraction', 'loss', 'seed', 'train_all', 'legacy_amp')}
    start_epoch, best = 0, -1.0
    if args.resume:
        ckpt = torch.load(out / 'last.pt', map_location='cpu', weights_only=True)
        if ckpt['config'] != config:
            raise ValueError('resume configuration mismatch')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch, best = ckpt['epoch'], ckpt['best']
        rng_restore(ckpt['rng'])
    print(json.dumps({'stage': 'train_start', 'config': config, 'train_images': len(train), 'dev_images': len(dev),
                      'amp_dtype': str(dtype), 'grad_scaler': scaler.is_enabled(), 'pretrained_coverage': coverage}), flush=True)
    torch.cuda.reset_peak_memory_stats()
    started, epoch = time.monotonic(), start_epoch - 1
    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses = []
        epoch_start = time.monotonic()
        for x, y in loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=dtype):
                logits, aux = forward(model, x)
                loss = loss_for(logits, y, expert, args.loss)
                if aux is not None:
                    loss = loss + 0.4 * loss_for(aux, y, expert, args.loss)
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite loss')
            scaler.scale(loss).backward()
            if not losses:
                scaler.unscale_(optimizer)
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                if not grads or not all(torch.isfinite(g).all().item() for g in grads):
                    raise FloatingPointError('missing or non-finite gradients')
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        scheduler.step()
        should_eval = bool(dev) and (epoch == 0 or (epoch+1) % 4 == 0 or epoch+1 == args.epochs)
        metrics = evaluate_subset(model, dev, args.mode, expert, args.smoke) if should_eval else None
        row = {'epoch': epoch+1, 'loss': sum(losses)/len(losses), 'dev': metrics,
               'epoch_seconds': round(time.monotonic()-epoch_start, 2),
               'peak_gpu_bytes': torch.cuda.max_memory_allocated()}
        with (out / 'metrics.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps(row, allow_nan=False), flush=True)
        if args.train_all or (metrics is not None and metrics['miou'] > best):
            best = metrics['miou'] if metrics is not None else -1.0
            save_atomic({'model_name': args.model, 'model': model.state_dict(), 'epoch': epoch+1,
                         'score': best, 'config': config}, out / 'best.pt')
        save_atomic({'model_name': args.model, 'model': model.state_dict(), 'epoch': epoch+1, 'best': best,
                     'config': config, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                     'scaler': scaler.state_dict(), 'rng': rng_save()}, out / 'last.pt')
        if args.smoke or time.monotonic()-started >= args.max_seconds:
            break
    if not (out / 'best.pt').exists():
        raise RuntimeError('training produced no evaluated checkpoint')
    atomic_json(out / 'summary.json', {'config': config, 'epochs_completed': epoch+1, 'best_score': best,
                'score_is_human_iou': expert, 'elapsed_seconds': round(time.monotonic()-started, 2),
                'peak_gpu_bytes': torch.cuda.max_memory_allocated(), 'budget_stop': epoch+1 < args.epochs})


if __name__ == '__main__':
    main()
