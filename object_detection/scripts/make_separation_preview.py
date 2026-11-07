"""Draw a small contact sheet of separation labels for human review."""
import json
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'separation_dataset_1000'
OUT = ROOT / 'reports' / '03_专业数据集核查' / 'separation_label_preview.jpg'
COLORS = [(40, 180, 40), (40, 100, 220)]
NAMES = ['component', 'flash/smoke']


def main():
    manifest = json.loads((DATA / 'manifest.json').read_text(encoding='utf-8'))
    chosen = [x for x in manifest if x['image'].endswith(('_000.jpg', '_020.jpg', '_040.jpg'))][:12]
    tiles = []
    for item in chosen:
        image_path = DATA / item['image']
        image = cv2.imdecode(__import__('numpy').fromfile(str(image_path), dtype=__import__('numpy').uint8), cv2.IMREAD_COLOR)
        if image is None:
            continue
        h, w = image.shape[:2]
        label_path = DATA / item['label']
        for row in label_path.read_text(encoding='utf-8').splitlines():
            bits = row.split(); cls = int(bits[0]); cx, cy, bw, bh = map(float, bits[1:])
            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            cv2.rectangle(image, (x1, y1), (x2, y2), COLORS[cls], 3)
            cv2.putText(image, NAMES[cls], (max(0, x1), max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, .7, COLORS[cls], 2, cv2.LINE_AA)
        cv2.putText(image, f"{item['alias']} {item['timestamp_seconds']:.1f}s", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2, cv2.LINE_AA)
        image = cv2.resize(image, (400, 240))
        tiles.append(image)
    while len(tiles) < 12:
        tiles.append(__import__('numpy').zeros((240, 400, 3), dtype='uint8'))
    rows = [__import__('numpy').hstack(tiles[i:i + 4]) for i in range(0, 12, 4)]
    sheet = __import__('numpy').vstack(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode('.jpg', sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError('Could not encode preview')
    OUT.write_bytes(encoded.tobytes())
    print(OUT)


if __name__ == '__main__':
    main()
