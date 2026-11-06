#!/usr/bin/env python3
"""Create paired, offline image-mask augmentations for train rows only."""
import argparse
import csv
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageEnhance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--output-manifest', type=Path, required=True)
    p.add_argument('--target-ratio', type=float, default=0.75,
                   help='Minority target as fraction of largest train class (default: 0.75).')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--rotation-deg', type=float, default=15)
    p.add_argument('--brightness', type=float, default=0.15)
    p.add_argument('--contrast', type=float, default=0.15)
    a = p.parse_args()
    if not 0 < a.target_ratio <= 1:
        p.error('--target-ratio must be in (0, 1].')
    with a.manifest.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
        fields = list(rows[0]) if rows else []
    if not rows:
        raise ValueError('Manifest is empty.')
    train = [r for r in rows if r['split'] == 'train']
    if not train:
        raise ValueError('Manifest has no train rows.')
    counts = Counter(int(r['label_index']) for r in train)
    max_count = max(counts.values())
    targets = {c: max(n, int(max_count * a.target_ratio + 0.5)) for c, n in counts.items()}
    rng = random.Random(a.seed)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    additions = []
    for cls, count in counts.items():
        source_rows = [r for r in train if int(r['label_index']) == cls]
        for j in range(targets[cls] - count):
            row = dict(rng.choice(source_rows))
            angle = rng.uniform(-a.rotation_deg, a.rotation_deg)
            flip = rng.random() < 0.5
            stem = Path(row['filename']).stem if row.get('filename') else Path(row['image_path']).stem
            suffix = f'__offaug_c{cls}_{j:05d}'
            image_out = a.output_dir / f'{stem}{suffix}.jpg'
            mask_out = a.output_dir / f'{stem}{suffix}_mask.png'
            with Image.open(row['image_path']) as im:
                im = im.convert('RGB').rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(0,0,0))
                if flip:
                    im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                im = ImageEnhance.Brightness(im).enhance(rng.uniform(1-a.brightness, 1+a.brightness))
                im = ImageEnhance.Contrast(im).enhance(rng.uniform(1-a.contrast, 1+a.contrast))
                im.save(image_out, quality=95)
            with Image.open(row['mask_path']) as m:
                m = m.convert('L').rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=0)
                if flip:
                    m = m.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                m.save(mask_out)
            row['image_path'] = str(image_out.resolve())
            if 'preprocessed_path' in row:
                row['preprocessed_path'] = str(image_out.resolve())
            row['mask_path'] = str(mask_out.resolve())
            if row.get('filename'):
                row['filename'] = f'{stem}{suffix}.jpg'
            additions.append(row)
    output_rows = rows + additions
    a.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with a.output_manifest.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    after = Counter(int(r['label_index']) for r in output_rows if r['split'] == 'train')
    print('Original train counts:', dict(sorted(counts.items())))
    print('Augmented train counts:', dict(sorted(after.items())))
    print('Validation/test rows copied unchanged:', len(output_rows)-len(additions)-len(train))
    print('New train pairs:', len(additions))
    print('Manifest:', a.output_manifest)

if __name__ == '__main__':
    main()
