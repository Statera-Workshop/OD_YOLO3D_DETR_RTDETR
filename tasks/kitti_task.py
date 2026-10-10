"""
KITTI BEV tasks: Complex-YOLO ("complex") and YOLO3D ("yolo3d").

Both tasks share the dataset, BEV map, NMS, decoding, evaluation and demos.
Only the raw-output layout differs:

    complex: x y w l im re        conf cls...   (z, h fixed per class)
    yolo3d : x y z h w l im re    conf cls...

Detection contract (metric LiDAR frame, identical for both tasks):
    {"boxes": [N, 7] = x, y, z, h, w, l, yaw   (z = box bottom)
     "scores": [N], "labels": [N],
     "bev_px": [N, 6] = x, y, w, l, im, re      (BEV pixels, for drawing)}
"""

import time

import numpy as np
import torch

from common import config as C
from common.datasets.kitti_bev import KittiBEVDataset
from common.geometry import bev_iou_3dboxes, iou_3d, nms_rotated
from common.kitti import config as cnf
from common.kitti.kitti_bev_utils import get_corners
from common.progress import Progress
from common.metrics import evaluate_detections, print_results
from common.trainer import make_loader

LAYOUT = {
    #          BEV (x, y, w, l, im, re) cols   conf col
    "yolo3d":  ((0, 1, 4, 5, 6, 7), 8),
}


class KittiTask:
    def __init__(self, name):
        self.name = name
        self.spec = C.TASKS[name]
        self.class_names = list(self.spec["classes"])
        self.bev_cols, self.conf_col = LAYOUT[name]

    # ------------------------------------------------------------ data
    def train_set(self, args):
        return KittiBEVDataset(C.KITTI_ROOT, self.spec["train_split"], self.name, train=True,
                               hflip_prob=C.PROFILE["hflip_prob"], max_samples=args.max_train)

    def test_set(self, args):
        return KittiBEVDataset(C.KITTI_ROOT, self.spec["test_split"], self.name, train=False,
                               max_samples=args.max_test)

    # ------------------------------------------------------------ post-process
    def decode(self, rows, labels, size):
        """BEV-pixel predictions -> metric LiDAR boxes (inverse of the target encoders)."""
        bc = cnf.boundary
        dx, dy, dz = bc["maxX"] - bc["minX"], bc["maxY"] - bc["minY"], bc["maxZ"] - bc["minZ"]
        boxes = np.zeros((len(rows), 7))
        for k, (r, c) in enumerate(zip(rows, labels)):
            bx, by, bw, bl, im, re = r[list(self.bev_cols)]
            x = by / size * dx + bc["minX"]
            y = bx / size * dy + bc["minY"]
            w = bw / size * dy - cnf.BOX_WL_MARGIN
            l = bl / size * dx - cnf.BOX_WL_MARGIN
            yaw = -np.arctan2(im, re)
            z = r[2] * dz + bc["minZ"]
            h = r[3] / size * dz
            boxes[k] = (x, y, z, h, max(w, 0.1), max(l, 0.1), yaw)
        return boxes

    def postprocess(self, raw, size, conf_thr):
        P = C.PROFILE
        out = []
        for pred in raw:
            pred = pred.detach().cpu()
            cls_conf, cls_id = pred[:, self.conf_col + 1:].max(1)
            score = pred[:, self.conf_col] * cls_conf
            keep = score >= conf_thr
            pred, score, cls_id = pred[keep], score[keep], cls_id[keep]
            if score.numel() > P["pre_nms_topk"]:
                top = score.topk(P["pre_nms_topk"]).indices
                pred, score, cls_id = pred[top], score[top], cls_id[top]
            pred, score, cls_id = pred.numpy().astype(np.float64), score.numpy(), cls_id.numpy()
            if len(pred):
                b = pred[:, list(self.bev_cols)]
                corners = np.stack([get_corners(x, y, w, l, np.arctan2(im, re)) for x, y, w, l, im, re in b])
                k = nms_rotated(corners, score, cls_id, P["nms_iou"])[: P["max_det"]]
            else:
                k = np.zeros(0, dtype=np.int64)
            pred, score, cls_id = pred[k], score[k], cls_id[k]
            out.append({"boxes": self.decode(pred, cls_id, size), "scores": score, "labels": cls_id,
                        "bev_px": pred[:, list(self.bev_cols)] if len(pred) else np.zeros((0, 6))})
        return out

    # ------------------------------------------------------------ evaluation
    @torch.no_grad()
    def evaluate(self, model, dataset, args, device):
        model.eval()
        loader = make_loader(dataset, args.batch_size, args.workers, False, args.seed)
        dets, gts = [], []
        t0 = time.time()
        bar = Progress(len(loader), f"{args.run_name} | test inference")
        for bi, (sids, bevs, _) in enumerate(loader, 1):
            raw = model(bevs.to(device, dtype=torch.float32))
            dets += self.postprocess(raw, bevs.shape[-1], C.PROFILE["eval_conf"])
            gts += [dataset.ground_truth(s) for s in sids]
            bar.update()
        bar.close()
        fps = len(dataset) / max(time.time() - t0, 1e-9)
        thr = C.PROFILE["eval_iou"]
        bev = evaluate_detections(dets, gts, self.class_names, bev_iou_3dboxes, thr)
        box3d = evaluate_detections(dets, gts, self.class_names, iou_3d, thr)
        print_results(f"KITTI_5k {self.spec['test_split']} BEV ({len(dataset)} frames)", bev)
        print_results(f"KITTI_5k {self.spec['test_split']} 3D  ({len(dataset)} frames)", box3d)
        print(f"Inference speed incl. post-process: {fps:.1f} frames/s")
        return {"metric": "KITTI BEV/3D AP50 (single difficulty, LiDAR frame)",
                "frames": len(dataset), "fps": fps, "bev_ap50": bev, "3d_ap50": box3d}
