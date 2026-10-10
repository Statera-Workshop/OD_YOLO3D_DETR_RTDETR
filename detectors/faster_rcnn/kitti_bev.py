"""
Faster R-CNN on the KITTI BEV tracks (Complex-YOLO: x, y, w, l, yaw; YOLO3D: + z, h) - 2-Stage reference.

The 1-Stage KITTI rule (backbone + neck with the YOLOv3 layers) would remove the RPN and the RoI head, so the 2-Stage
track keeps Faster R-CNN and adds a rotated-box branch to the RoI head (R2CNN-style, Jiang et al., arXiv 1706.09579):
    RPN and RoI sampling work on the axis-aligned box enclosing each rotated GT box (torchvision Faster R-CNN, unchanged);
    the RoI head predicts, class-agnostically, the rotated box relative to its RoI:
        dx = (x - px) / pw, dy = (y - py) / ph, dw = log(w / pw), dl = log(l / ph), (im, re) = (sin, cos) of the yaw,
        YOLO3D also z (metres, direct) and dh = log(h / pw).
    Loss: smooth L1 (beta 1/9) on the deltas + L1 on (im, re) + the Complex-YOLO unit-norm penalty (1 - im^2 - re^2)^2,
    over the positive RoIs, / #sampled RoIs (the torchvision box-loss normalisation).
Targets and outputs follow the YOLOv3 KITTI layers of this framework (detectors/yolov3): target rows
[b, cls, x, y, w, l, im, re] (complex) / [b, cls, x, y, z, h, w, l, im, re] (yolo3d), x, y, w, l, h normalised by the
image size; output rows [x, y, w, l, im, re, conf, cls...] / [x, y, z, h, w, l, im, re, conf, cls...] in pixels, so the
KITTI evaluation and post-processing of the framework are used unchanged. conf = 1, cls = softmax foreground probabilities.
"""

import torch


NUM_ATTR = {"complex": 6, "yolo3d": 8}          # x y w l im re  /  x y z h w l im re


def rotated_to_aabb(x, y, w, l, im, re):
    """axis-aligned box enclosing the rotated rectangle (half extents w / 2, l / 2 rotated by atan2(im, re))"""
    n = torch.sqrt(im ** 2 + re ** 2).clamp(min=1e-6)
    s, c = (im / n).abs(), (re / n).abs()
    hx = (w * c + l * s) / 2
    hy = (w * s + l * c) / 2
    return torch.stack([x - hx, y - hy, x + hx, y + hy], -1)
