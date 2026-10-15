"""Camera image -> policy input. The SAME function runs when building the dataset
and on the robot, so training and deployment see identical pixels.

A spec says how one camera's 640x480 RGB frame becomes size x size:
    resize               squash the whole frame (keeps the full view, distorts aspect)
    center_crop          largest centered square, then resize
    crop:x,y,w,h         this pixel rectangle (of the original frame), then resize
    pad                  letterbox to a square with black bars, then resize
"""

import cv2
import numpy as np


def parse(spec):
    spec = spec.strip()
    if spec in ('resize', 'center_crop', 'pad'):
        return (spec,)
    if spec.startswith('crop:'):
        values = [int(v) for v in spec[5:].split(',')]
        if len(values) != 4 or values[2] <= 0 or values[3] <= 0:
            raise ValueError('crop spec is crop:x,y,w,h, got %r' % spec)
        return ('crop', *values)
    raise ValueError('unknown image spec %r (resize, center_crop, crop:x,y,w,h, pad)' % spec)


def apply(rgb, spec, size):
    """rgb: H x W x 3 uint8 (RGB order) -> size x size x 3 uint8."""
    kind = parse(spec) if isinstance(spec, str) else spec
    h, w = rgb.shape[:2]
    if kind[0] == 'center_crop':
        side = min(h, w)
        y, x = (h - side) // 2, (w - side) // 2
        rgb = rgb[y:y + side, x:x + side]
    elif kind[0] == 'crop':
        x, y, cw, ch = kind[1:]
        if x < 0 or y < 0 or x + cw > w or y + ch > h:
            raise ValueError('crop %s is outside the %dx%d frame' % (kind[1:], w, h))
        rgb = rgb[y:y + ch, x:x + cw]
    elif kind[0] == 'pad':
        side = max(h, w)
        canvas = np.zeros((side, side, 3), dtype=rgb.dtype)
        y, x = (side - h) // 2, (side - w) // 2
        canvas[y:y + h, x:x + w] = rgb
        rgb = canvas
    return cv2.resize(np.ascontiguousarray(rgb), (size, size), interpolation=cv2.INTER_AREA)
