"""Independently validate one checkpoint, preserving exact per-class metrics."""
import argparse
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(str(ROOT))
os.environ.setdefault('GIT_PYTHON_REFRESH', 'quiet')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--img', type=int, default=640)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--device', default='0')
    parser.add_argument('--output', required=True, help='New result directory')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--max-det', type=int, default=300)
    args = parser.parse_args()
    import torch
    import yaml
    import val
    from utils.callbacks import Callbacks
    from utils.general import LOGGER

    torch.set_num_threads(4)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    callbacks = Callbacks()
    captured = {}
    validated_paths = []
    data = yaml.safe_load(Path(args.data).read_text(encoding='utf-8'))
    names = data['names']

    def capture(nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix):
        captured['instances'] = int(nt.sum())
        captured['per_class'] = [
            dict(id=int(c), name=names[int(c)], instances=int(nt[int(c)]),
                 precision=float(p[i]), recall=float(r[i]), f1=float(f1[i]),
                 ap50=float(ap50[i]), ap50_95=float(ap[i]))
            for i, c in enumerate(ap_class)]
        captured['confusion_matrix'] = confusion_matrix.matrix.tolist()

    def capture_batch(batch_i, im, targets, paths, shapes, preds):
        validated_paths.extend(str(Path(p).resolve()) for p in paths)

    callbacks.register_action('on_val_end', name='capture_metrics', callback=capture)
    callbacks.register_action('on_val_batch_end', name='capture_image_count', callback=capture_batch)
    with (output / 'validation.log').open('w', encoding='utf-8') as stream:
        handler = logging.StreamHandler(stream)
        LOGGER.addHandler(handler)
        try:
            metrics, _, speed = val.run(
                weights=Path(args.weights), data=args.data, imgsz=args.img,
                batch_size=args.batch, device=args.device, workers=0,
                half=args.device != 'cpu', conf_thres=0.001, iou_thres=0.6,
                max_det=args.max_det, augment=args.augment, verbose=True,
                plots=True, callbacks=callbacks, project=output, name='plots', exist_ok=True)
        finally:
            LOGGER.removeHandler(handler)
    report = dict(
        timestamp=datetime.now().astimezone().isoformat(),
        command=sys.argv, weights=str(Path(args.weights).resolve()),
        weights_sha256=sha256(args.weights), data=args.data,
        images=len(validated_paths),
        validated_paths_sha256=hashlib.sha256('\n'.join(sorted(validated_paths)).encode('utf-8')).hexdigest(),
        data_yaml_sha256=sha256(args.data), image_size=args.img,
        batch_size=args.batch, half=args.device != 'cpu',
        confidence_threshold=0.001, nms_iou_threshold=0.6,
        max_detections=args.max_det, augmented_inference=args.augment,
        precision=float(metrics[0]), recall=float(metrics[1]),
        map50=float(metrics[2]), map50_95=float(metrics[3]),
        speed_ms_per_image=list(speed), **captured)
    (output / 'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
