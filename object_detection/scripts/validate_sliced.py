"""Whole-image plus overlapping original-image crops; use the unchanged YOLO AP evaluator.

The inference callback receives only pixels, paths and resize metadata. It never
receives labels. Targets are consumed solely by the existing val.py evaluator.
This is a more expensive inference protocol, reported separately from 640
whole-image inference. It is inspired by SAHI; it does not import the SAHI package.
"""
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
os.environ.setdefault('GIT_PYTHON_REFRESH', 'quiet')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def positions(length, size, overlap):
    if length <= size:
        return [0]
    step = max(1, int(round(size * (1 - overlap))))
    return sorted(set(list(range(0, length - size + 1, step)) + [length - size]))


class SlicedInference:
    def __init__(self, size, tile_size, overlap, tile_batch, edge_margin, whole_only=False):
        self.size, self.tile_size, self.overlap = size, tile_size, overlap
        self.tile_batch, self.edge_margin = tile_batch, edge_margin
        self.whole_only = whole_only
        self.records = []

    def __call__(self, model, im, paths, shapes, augment):
        import cv2
        import numpy as np
        import torch
        from utils.augmentations import letterbox
        from utils.general import scale_boxes, xywh2xyxy, xyxy2xywh

        global_predictions = model(im, augment=augment)
        if isinstance(global_predictions, (tuple, list)):
            global_predictions = global_predictions[0]
        outputs = []
        for index, path in enumerate(paths):
            raw = global_predictions[index]
            parts = [raw[raw[:, 4] > .001]]
            image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError('Cannot decode ' + str(path))
            height, width = image.shape[:2]
            boxes = []
            if not self.whole_only and max(height, width) > self.tile_size:
                for y in positions(height, self.tile_size, self.overlap):
                    for x in positions(width, self.tile_size, self.overlap):
                        boxes.append((x, y, min(x + self.tile_size, width), min(y + self.tile_size, height)))
            for start in range(0, len(boxes), self.tile_batch):
                arrays, metadata = [], []
                for x1, y1, x2, y2 in boxes[start:start + self.tile_batch]:
                    crop = image[y1:y2, x1:x2]
                    resized, ratio, pad = letterbox(crop, self.size, auto=False, stride=32)
                    arrays.append(np.ascontiguousarray(resized[:, :, ::-1].transpose(2, 0, 1)))
                    metadata.append((x1, y1, x2, y2, crop.shape[:2], ratio, pad))
                tensor = torch.from_numpy(np.stack(arrays)).to(im.device).type_as(im) / 255
                predictions = model(tensor, augment=augment)
                if isinstance(predictions, (tuple, list)):
                    predictions = predictions[0]
                for prediction, meta in zip(predictions, metadata):
                    x1, y1, x2, y2, crop_shape, ratio, pad = meta
                    prediction = prediction[prediction[:, 4] > .001].clone()
                    if not len(prediction):
                        continue
                    xyxy = xywh2xyxy(prediction[:, :4])
                    scale_boxes(tensor.shape[2:], xyxy, crop_shape, (ratio, pad))
                    # Interior crop borders can cut a large object. The global
                    # view retains it; avoid treating truncated fragments as new bodies.
                    keep = torch.ones(len(xyxy), dtype=torch.bool, device=im.device)
                    margin = self.edge_margin
                    if x1 > 0:
                        keep &= xyxy[:, 0] > margin
                    if y1 > 0:
                        keep &= xyxy[:, 1] > margin
                    if x2 < width:
                        keep &= xyxy[:, 2] < crop_shape[1] - margin
                    if y2 < height:
                        keep &= xyxy[:, 3] < crop_shape[0] - margin
                    xyxy, prediction = xyxy[keep], prediction[keep]
                    xyxy[:, [0, 2]] += x1
                    xyxy[:, [1, 3]] += y1
                    # Original-image coordinates -> current val batch letterbox.
                    gain, batch_pad = shapes[index][1]
                    xyxy[:, [0, 2]] = xyxy[:, [0, 2]] * gain[0] + batch_pad[0]
                    # val.py's scale_boxes uses gain[0] for both axes. Match its
                    # inverse exactly, including rounded rectangular resizing.
                    xyxy[:, [1, 3]] = xyxy[:, [1, 3]] * gain[0] + batch_pad[1]
                    prediction[:, :4] = xyxy2xywh(xyxy)
                    parts.append(prediction)
            combined = torch.cat(parts, 0)
            outputs.append(combined)
            self.records.append(dict(path=str(Path(path).resolve()), width=width, height=height,
                                     crops=len(boxes), candidates_before_nms=len(combined)))
        result = im.new_zeros((len(outputs), max(len(p) for p in outputs), global_predictions.shape[2]))
        for index, prediction in enumerate(outputs):
            result[index, :len(prediction)] = prediction
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--data', default='yolov5_project.yaml')
    parser.add_argument('--output', required=True)
    parser.add_argument('--img', type=int, default=640)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--tile-size', type=int, default=640)
    parser.add_argument('--overlap', type=float, default=.2)
    parser.add_argument('--tile-batch', type=int, default=2)
    parser.add_argument('--edge-margin', type=float, default=2.)
    parser.add_argument('--device', default='0')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--whole-only', action='store_true', help='Control: no crops, same callback and AP evaluation')
    args = parser.parse_args()
    if not 0 <= args.overlap < 1 or args.tile_size < 32 or args.tile_batch < 1:
        parser.error('Invalid slicing configuration')
    os.chdir(str(ROOT))
    import torch
    import yaml
    import val
    from utils.callbacks import Callbacks
    from utils.general import LOGGER
    torch.set_num_threads(4)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    names = yaml.safe_load(Path(args.data).read_text(encoding='utf-8'))['names']
    captured = {}
    callbacks = Callbacks()

    def capture(nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix):
        captured['instances'] = int(nt.sum())
        captured['per_class'] = [dict(id=int(c), name=names[int(c)], instances=int(nt[int(c)]),
            precision=float(p[i]), recall=float(r[i]), f1=float(f1[i]), ap50=float(ap50[i]),
            ap50_95=float(ap[i])) for i, c in enumerate(ap_class)]
        captured['confusion_matrix'] = confusion_matrix.matrix.tolist()

    callbacks.register_action('on_val_end', name='capture_metrics', callback=capture)
    inference = SlicedInference(args.img, args.tile_size, args.overlap, args.tile_batch,
                                args.edge_margin, args.whole_only)
    with (output / 'validation.log').open('w', encoding='utf-8') as stream:
        handler = logging.StreamHandler(stream)
        LOGGER.addHandler(handler)
        try:
            metrics, _, speed = val.run(weights=Path(args.weights), data=args.data,
                imgsz=args.img, batch_size=args.batch, device=args.device, workers=0,
                half=args.device != 'cpu', conf_thres=.001, iou_thres=.6, max_det=300,
                augment=args.augment, verbose=True, plots=True, callbacks=callbacks,
                project=output, name='plots', exist_ok=True, inference_callback=inference)
        finally:
            LOGGER.removeHandler(handler)
    paths = [row['path'] for row in inference.records]
    record = dict(timestamp=datetime.now().astimezone().isoformat(), command=sys.argv,
        weights=str(Path(args.weights).resolve()), weights_sha256=digest(args.weights),
        data=args.data, data_yaml_sha256=digest(args.data), images=len(paths),
        validated_paths_sha256=hashlib.sha256('\n'.join(sorted(paths)).encode('utf-8')).hexdigest(),
        image_size=args.img, batch_size=args.batch, half=args.device != 'cpu',
        confidence_threshold=.001, nms_iou_threshold=.6, max_detections=300,
        augmented_inference=args.augment, inference_method='whole_only_control' if args.whole_only else 'whole_plus_overlapping_slices',
        slicing=dict(tile_size_original_pixels=args.tile_size, overlap=args.overlap,
                     tile_batch=args.tile_batch, edge_margin=args.edge_margin,
                     total_crops=sum(row['crops'] for row in inference.records)),
        precision=float(metrics[0]), recall=float(metrics[1]), map50=float(metrics[2]),
        map50_95=float(metrics[3]), speed_ms_per_image=list(speed), **captured,
        scope='Original unchanged validation images/labels, standard AP matching; more image views and computation than whole-image inference. No labels passed to inference.')
    (output / 'metrics.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'inference_views.json').write_text(json.dumps(inference.records, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(record, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
