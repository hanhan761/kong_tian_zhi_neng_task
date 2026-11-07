"""Freeze actual annotation bytes and image hashes used by a training run."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(data, output):
    data, output = Path(data).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    files, counts = [], {}
    for split in ['train', 'val', 'test']:
        images = sorted((data / 'images' / split).glob('*.jpg'))
        counts[split] = len(images)
        for image in images:
            label = data / 'labels' / split / (image.stem + '.txt')
            if not label.is_file():
                raise FileNotFoundError(label)
            files.append(dict(image=image.relative_to(data).as_posix(), image_sha256=sha(image),
                              label=label.relative_to(data).as_posix(), label_sha256=sha(label)))
    if (data / 'manifest.json').exists():
        manifest = json.loads((data / 'manifest.json').read_text(encoding='utf-8'))
        expected = {item['image']: item for item in manifest}
        for item in files:
            for key in ['image_sha256', 'label_sha256']:
                if key in expected[item['image']] and item[key] != expected[item['image']][key]:
                    raise ValueError('Dataset differs from manifest: ' + item['image'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(data.glob('*')):
            if path.is_file() and path.suffix in ['.json', '.yaml', '.txt']:
                archive.write(path, path.name)
        for path in sorted((data / 'labels').rglob('*.txt')):
            archive.write(path, path.relative_to(data).as_posix())
        archive.writestr('file_hashes.json', json.dumps(files, ensure_ascii=False, indent=2))
    result = dict(created=datetime.now().astimezone().isoformat(), dataset=str(data),
                  archive=str(output), archive_sha256=sha(output), splits=counts,
                  images=sum(counts.values()),
                  note='Contains labels, metadata and image hashes; original images stay in dataset/images.')
    output.with_suffix('.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(ROOT / args.data, ROOT / args.output), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
