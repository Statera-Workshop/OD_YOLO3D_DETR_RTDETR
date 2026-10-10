"""
RT-DETR builder (RT-DETR-R50: PResNet-50-vd, hybrid encoder, 6-layer decoder, 300 queries), Image task.

Paper: Zhao et al., "DETRs Beat YOLOs on Real-time Object Detection" (CVPR 2024, arXiv 2304.08069).
Code : official lyuwenyu/RT-DETR, rtdetr_pytorch (configs/rtdetr/rtdetr_r50vd_6x_coco.yml + include/rtdetr_r50vd.yml),
       files copied unchanged into detectors/rtdetr/official/ except two imports (src.core.register and src.misc.dist
       -> official/core_min.py): rtdetr.py, presnet.py + common.py (PResNet depth 50, variant d, return_idx [1, 2, 3]),
       hybrid_encoder.py (AIFI on S5, CCFM, hidden 256, FFN 1024, gelu / silu), rtdetr_decoder.py (deformable
       cross-attention in PyTorch, 6 layers, 300 queries, IoU-aware query selection, 100 denoising queries),
       denoising.py, matcher.py (focal costs class 2, bbox 5, giou 2), rtdetr_criterion.py (varifocal 1, L1 5, GIoU 2,
       alpha 0.75, gamma 2, auxiliary + denoising losses), box_ops.py, utils.py.

Raw contract: 300 queries per image, sigmoid class probabilities (objectness 1); RT-DETR is NMS-free, the common
post-processing still applies its NMS ([Unified]).

[Unified] no ImageNet weights (pretrained False), trainable BN (freeze_norm False; FrozenBN in the original) +
PreciseBN, common init (nn.Conv2d / BN), no mean / std, single 608 scale (multi_scale off, eval_spatial_size 608),
common augmentation, Adam 1e-4 instead of AdamW 1e-4 / backbone 1e-5 + EMA; gradient clipping 0.1 as the official
recipe (common/trainer.py hook), 30 epochs (official 72), common post-processing.
"""

import torch
import torch.nn as nn

from detectors.anchor_free.common import HeadStats, split_targets, to_raw
from detectors.detr.official.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from detectors.rtdetr.official.hybrid_encoder import HybridEncoder
from detectors.rtdetr.official.matcher import HungarianMatcher
from detectors.rtdetr.official.presnet import PResNet
from detectors.rtdetr.official.rtdetr import RTDETR
from detectors.rtdetr.official.rtdetr_criterion import SetCriterion
from detectors.rtdetr.official.rtdetr_decoder import RTDETRTransformer


class RTDETRDet(nn.Module):
    def __init__(self, num_classes, img_size=608, depth=50):
        super().__init__()
        self.num_classes, self.num_queries = num_classes, 300
        backbone = PResNet(depth=depth, variant="d", num_stages=4, return_idx=[1, 2, 3], freeze_at=-1,
                           freeze_norm=False, pretrained=False)
        encoder = HybridEncoder(in_channels=[512, 1024, 2048], feat_strides=[8, 16, 32], hidden_dim=256,
                                use_encoder_idx=[2], num_encoder_layers=1, nhead=8, dim_feedforward=1024, dropout=0.0,
                                enc_act="gelu", pe_temperature=10000, expansion=1.0, depth_mult=1, act="silu",
                                eval_spatial_size=[img_size, img_size])
        decoder = RTDETRTransformer(num_classes=num_classes, feat_channels=[256, 256, 256], feat_strides=[8, 16, 32],
                                    hidden_dim=256, num_levels=3, num_queries=300, num_decoder_layers=6,
                                    num_denoising=100, eval_idx=-1, eval_spatial_size=[img_size, img_size])
        self.model = RTDETR(backbone, encoder, decoder, multi_scale=None)
        matcher = HungarianMatcher(weight_dict={"cost_class": 2, "cost_bbox": 5, "cost_giou": 2},
                                   use_focal_loss=True, alpha=0.25, gamma=2.0)
        self.criterion = SetCriterion(matcher, weight_dict={"loss_vfl": 1, "loss_bbox": 5, "loss_giou": 2},
                                      losses=["vfl", "boxes"], alpha=0.75, gamma=2.0, num_classes=num_classes)
        self.clip_max_norm = 0.1                       # official optimizer.yml clip_max_norm (common/trainer.py hook)
        self.precise_bn, self.precise_bn_images, self.precise_bn_batch_size = True, 1600, 16
        self.stats = HeadStats()
        self.yolo_layers = [self.stats]

    def forward(self, x, targets=None):
        B, C, S = x.shape[0], self.num_classes, x.shape[2]
        if targets is None and self.training:                                   # PreciseBN (BN: backbone, encoder)
            self.model.encoder(self.model.backbone(x))
            return x.new_zeros(B, 0, 5 + C)
        if targets is None:
            out = self.model(x)
            return to_raw(box_cxcywh_to_xyxy(out["pred_boxes"]) * S, out["pred_logits"].sigmoid())
        tgt = [{"labels": t["labels"], "boxes": box_xyxy_to_cxcywh(t["boxes"] / S)}
               for t in split_targets(targets, B, S)]
        out = self.model(x, targets=tgt)
        loss_dict = self.criterion(out, tgt)                                   # already weighted
        loss = sum(loss_dict.values())
        with torch.no_grad():
            idx = self.criterion.matcher({"pred_logits": out["pred_logits"], "pred_boxes": out["pred_boxes"]}, tgt)
            fg = torch.zeros(B, out["pred_logits"].shape[1], dtype=torch.bool, device=x.device)
            lab = torch.full_like(fg, -1, dtype=torch.long)
            for b, (i, j) in enumerate(idx):
                fg[b, i] = True
                lab[b, i] = tgt[b]["labels"][j].to(x.device)
            logits = out["pred_logits"].detach().reshape(-1, C)
        self.stats.update({"losses": loss, "loss_vfl": loss_dict["loss_vfl"], "loss_bbox": loss_dict["loss_bbox"],
                           "loss_giou": loss_dict["loss_giou"], "num_pos": fg.float().sum() / B},
                          logits.sigmoid().max(-1).values, logits, fg.view(-1), lab.view(-1)[fg.view(-1)])
        return loss, x.new_zeros(B, 0, 5 + C)

    def load_darknet_weights(self, path):
        raise NotImplementedError("RT-DETR has no Darknet weights; train from scratch.")


def build_model(task, num_classes):
    if task in ("complex", "yolo3d"):
        from detectors.rtdetr.kitti import RTDETRKitti
        return RTDETRKitti(task, num_classes)
    raise ValueError(f"unknown task {task}")
