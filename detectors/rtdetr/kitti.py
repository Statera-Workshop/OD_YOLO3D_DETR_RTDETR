"""
RT-DETR on the KITTI BEV tracks (DETR-family KITTI reference, same design as detectors/deformable_detr/kitti.py).

The official decoder is used unchanged; a forward hook on its last decoder layer reads the query states (the
official module returns only boxes / logits), the denoising queries are split off with dn_meta['dn_num_split'] exactly
as RTDETRTransformer.forward does, and a 3-layer MLP predicts w, l (sigmoid, / image), (im, re) and for YOLO3D also
z (metres) and h (sigmoid, / image); the centre is the centre of the predicted axis-aligned box. Matching and the
official varifocal / L1 / GIoU losses work on the axis-aligned box enclosing each rotated GT; the rotated loss is the
one of detectors/deformable_detr/kitti.py. Outputs follow the Complex-YOLO / YOLO3D row layout (conf 1, sigmoid
class probabilities). KITTI lr: Adam 1e-4 (as all DETR-family models), gradient clipping 0.1.
"""

import torch
import torch.nn.functional as F

from detectors.deformable_detr.kitti import DeformableDETRKitti
from detectors.rtdetr.build import RTDETRDet
from detectors.rtdetr.official.rtdetr_decoder import MLP
from detectors.faster_rcnn.kitti_bev import NUM_ATTR


class RTDETRKitti(RTDETRDet):
    _gt = DeformableDETRKitti._gt
    _rot = DeformableDETRKitti._rot

    def __init__(self, task, num_classes, img_size=608):
        super().__init__(num_classes, img_size=img_size)
        self.task, self.n_rot = task, NUM_ATTR[task]
        self.rot_embed = MLP(256, 256, 6, 3)
        self._hs = None
        self.model.decoder.decoder.layers[-1].register_forward_hook(lambda m, i, o: setattr(self, "_hs", o))

    def forward(self, x, targets=None):
        B, C, S = x.shape[0], self.num_classes, x.shape[2]
        if targets is None and self.training:
            self.model.encoder(self.model.backbone(x))
            return x.new_zeros(B, 0, self.n_rot + 1 + C)
        if targets is None:
            out = self.model(x)
            rot = self._rot(self._hs)
            cxcy, wl, ang = out["pred_boxes"][..., :2] * S, rot[..., :2] * S, rot[..., 2:4]
            rows = torch.cat([cxcy, wl, ang], -1) if self.task == "complex" else \
                torch.cat([cxcy, rot[..., 4:5], rot[..., 5:6] * S, wl, ang], -1)
            probs = out["pred_logits"].sigmoid()
            return torch.cat([rows, torch.ones_like(probs[..., :1]), probs], -1)
        gts = self._gt(targets, B, S)
        tgt = [{"labels": g["labels"], "boxes": g["boxes"]} for g in gts]
        out = self.model(x, targets=tgt)
        hs = self._hs
        if "dn_meta" in out and out["dn_meta"] is not None:
            hs = torch.split(hs, out["dn_meta"]["dn_num_split"], dim=1)[1]
        rot = self._rot(hs)
        loss_dict = self.criterion(out, tgt)
        loss = sum(loss_dict.values())
        idx = self.criterion.matcher({"pred_logits": out["pred_logits"], "pred_boxes": out["pred_boxes"]}, tgt)
        src = torch.cat([rot[b, i] for b, (i, _) in enumerate(idx)])
        dst = torch.cat([gts[b]["rot"][j.to(x.device)] for b, (_, j) in enumerate(idx)])
        n = max(float(len(dst)), 1.0)
        if len(dst):
            cols = [0, 1, 4, 5]
            loss_rot = (5 * F.l1_loss(src[:, cols], dst[:, cols], reduction="sum")
                        + F.l1_loss(src[:, 2:4], dst[:, 2:4], reduction="sum")
                        + ((1 - (src[:, 2:4] ** 2).sum(1)) ** 2).sum()) / n
        else:
            loss_rot = rot.sum() * 0
        loss = loss + loss_rot
        with torch.no_grad():
            Q = out["pred_logits"].shape[1]
            fg = torch.zeros(B, Q, dtype=torch.bool, device=x.device)
            lab = torch.full((B, Q), -1, dtype=torch.long, device=x.device)
            for b, (i, j) in enumerate(idx):
                fg[b, i] = True
                lab[b, i] = tgt[b]["labels"][j].to(x.device)
            logits = out["pred_logits"].detach().reshape(-1, C)
        self.stats.update({"losses": loss, "loss_vfl": loss_dict["loss_vfl"], "loss_rot": loss_rot,
                           "num_pos": fg.float().sum() / B},
                          logits.sigmoid().max(-1).values, logits, fg.view(-1), lab.view(-1)[fg.view(-1)])
        return loss, x.new_zeros(B, 0, self.n_rot + 1 + C)
