#!/usr/bin/env python3
"""Side-by-side preview of image specs (franka/images.py) on recorded frames, to
choose how each camera becomes the policy's 128x128 input.

    python franka/preview_images.py <export>/<episode> --out preview.png \
        --spec resize --spec center_crop --spec pad --spec crop:240,60,400,400

Rows: camera x frame; columns: the original (with crop boxes) and each spec at
the policy size, enlarged so it can be judged by eye.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from franka import images  # noqa: E402


def read_frame(path, index):
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, bgr = capture.read()
    capture.release()
    if not ok:
        raise IOError('cannot read frame %d of %s' % (index, path))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def label(image, text):
    image = image.copy()
    cv2.putText(image, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('episode', help='an exported episode folder')
    parser.add_argument('--out', required=True)
    parser.add_argument('--spec', action='append', default=None, help='image spec (repeatable)')
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--frames', default='0.2,0.5,0.8', help='fractions of the episode to show')
    args = parser.parse_args()
    specs = args.spec or ['resize', 'center_crop', 'pad']
    folder = Path(args.episode)
    info = json.loads((folder / 'episode.json').read_text())
    count = info['frames']
    tile = 384
    rows = []
    for camera, cam_info in sorted(info['cameras'].items()):
        for fraction in [float(f) for f in args.frames.split(',')]:
            frame = read_frame(folder / cam_info['file'], int(fraction * (count - 1)))
            h, w = frame.shape[:2]
            overview = frame.copy()
            for spec in specs:
                kind = images.parse(spec)
                if kind[0] == 'center_crop':
                    side = min(h, w)
                    box = ((w - side) // 2, (h - side) // 2, side, side)
                elif kind[0] == 'crop':
                    box = kind[1:]
                else:
                    continue
                cv2.rectangle(overview, (box[0], box[1]), (box[0] + box[2] - 1, box[1] + box[3] - 1),
                              (255, 220, 0), 2)
            overview = cv2.resize(overview, (int(tile * w / h), tile), interpolation=cv2.INTER_AREA)
            tiles = [label(overview, '%s original %dx%d' % (camera, w, h))]
            for spec in specs:
                small = images.apply(frame, spec, args.size)
                tiles.append(label(cv2.resize(small, (tile, tile), interpolation=cv2.INTER_NEAREST),
                                   '%s @%d' % (spec, args.size)))
            rows.append(np.hstack(tiles))
    sheet = np.vstack(rows)
    cv2.imwrite(args.out, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print('wrote %s (%dx%d)' % (args.out, sheet.shape[1], sheet.shape[0]))


if __name__ == '__main__':
    main()
