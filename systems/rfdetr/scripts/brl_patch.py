#!/usr/bin/env python3
"""Stop punishing high-confidence unmatched queries - they are the missing labels.

Measured on 2026-08-04: 27% of column_base training images contain damage the
annotators did not box. Spalling the client demonstrably grades is missing on 32%
of occurrences, cracking on 47%. The same damage is therefore sometimes a
positive and sometimes background.

RF-DETR's classification loss makes this maximally harmful. In `loss_labels`
(ia_bce branch) every unmatched query carries

    neg_weights = prob ** gamma

so the *higher* the model's confidence, the *harder* it is pushed toward
background. That is the right behaviour when labels are complete - it suppresses
confident false positives - and exactly the wrong behaviour when they are not:
the queries the model is most sure about are precisely the unannotated damage,
and they receive the largest negative gradient. A detector trained this way
learns to assign middling scores to that damage type, which is the failure
measured all day (recall ceiling 0.875/0.940 against 0.514/0.590 usable).

Background Recalibration Loss (Zhang et al., "Solving Missing-Annotation Object
Detection with Background Recalibration Loss") addresses this by flipping the
gradient for background samples whose confidence exceeds a threshold, treating
them as positives. Reported gains: +7.2 mAP at 20.6% missing annotations, +6.4
at 39%, +10.7 at 65%, +9.0 on COCO with 50% removed. Our 27% sits inside that
range.

This patch implements the conservative half of that idea: unmatched queries above
the threshold have their negative weight **zeroed** rather than flipped, so they
are ignored instead of being asserted as objects. Flipping asserts that every
confident detection is a missing label, which is false here - the model also
produces genuine false positives, and the 2026-08-04 measurement puts its fire
rate on real sound elements at 95.7%. Zeroing removes the contradictory
supervision without inventing labels.

Applied by monkey-patching, because the loss lives inside the installed rfdetr
package and forking it for one term is not worth the maintenance.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from rfdetr.models import lwdetr
from rfdetr.util import box_ops

_ORIGINAL = lwdetr.SetCriterion.loss_labels
_THRESHOLD = 0.0
_MODE = "ignore"        # ignore | flip | pu


def set_brl(value: float, mode: str = "ignore") -> None:
    """0 disables the patch; typical thresholds 0.3-0.5.

    Three treatments of a confident-but-unmatched query, in increasing order of
    how strong a claim they make about it:

    ``ignore``  zero its background weight. Says only "we do not know whether this
                is a missing label", which is exactly what the audit established.
    ``flip``    the published BRL: train it as a positive. Says "this IS a missing
                label" - stronger, and wrong whenever the query is a genuine false
                positive, of which this model produces many (95.7% fire rate on
                unseen sound elements).
    ``pu``      positive-unlabeled framing: down-weight rather than remove, by the
                estimated share of unlabelled positives among the background. With
                27% of column_base images carrying unboxed damage, treating the
                background as ~0.75 reliable is closer to the truth than either
                1.0 (default) or 0.0 (ignore).
    """
    global _THRESHOLD, _MODE
    _THRESHOLD = float(value)
    _MODE = mode


def _loss_labels_brl(self, outputs, targets, indices, num_boxes, log=True):
    if _THRESHOLD <= 0 or not getattr(self, "ia_bce_loss", False):
        return _ORIGINAL(self, outputs, targets, indices, num_boxes, log=log)

    src_logits = outputs["pred_logits"]
    idx = self._get_src_permutation_idx(indices)
    target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

    alpha = self.focal_alpha
    gamma = 2
    src_boxes = outputs["pred_boxes"][idx]
    target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
    iou_targets = torch.diag(
        box_ops.box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes.detach()),
            box_ops.box_cxcywh_to_xyxy(target_boxes),
        )[0]
    )
    pos_ious = iou_targets.clone().detach()
    prob = src_logits.sigmoid()

    pos_weights = torch.zeros_like(src_logits)
    neg_weights = prob**gamma

    pos_ind = [id for id in idx]
    pos_ind.append(target_classes_o)

    t = prob[tuple(pos_ind)].pow(alpha) * pos_ious.pow(1 - alpha)
    t = torch.clamp(t, 0.01).detach()
    pos_weights[tuple(pos_ind)] = t.to(pos_weights.dtype)
    neg_weights[tuple(pos_ind)] = 1 - t.to(neg_weights.dtype)

    # --- the patch -------------------------------------------------------
    # Queries the model is confident about but the matcher left unmatched are
    # the candidates for missing annotations. Drop their background pressure.
    with torch.no_grad():
        suspect = prob > _THRESHOLD
        matched = torch.zeros_like(suspect)
        matched[tuple(pos_ind)] = True
        suspect = suspect & ~matched
    if _MODE == "ignore":
        neg_weights = neg_weights * (~suspect).to(neg_weights.dtype)
    elif _MODE == "flip":
        # published BRL: the suspect becomes a positive target
        neg_weights = neg_weights * (~suspect).to(neg_weights.dtype)
        pos_weights = pos_weights + suspect.to(pos_weights.dtype) * (1 - prob).pow(2).detach()
    elif _MODE == "pu":
        # keep some background pressure, scaled by how reliable the label is
        neg_weights = neg_weights * torch.where(
            suspect, torch.full_like(neg_weights, _PU_KEEP), torch.ones_like(neg_weights))
    # ---------------------------------------------------------------------

    loss_ce = neg_weights * src_logits - F.logsigmoid(src_logits) * (pos_weights + neg_weights)
    loss_ce = loss_ce.sum() / num_boxes
    losses = {"loss_ce": loss_ce}
    if log:
        losses["class_error"] = 100 - _accuracy(src_logits[idx], target_classes_o)[0]
    return losses


def _accuracy(output, target, topk=(1,)):
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100.0 / target.numel() for k in topk]


_PU_KEEP = 0.25          # residual background weight in `pu` mode


def apply(threshold: float, mode: str = "ignore", pu_keep: float = 0.25) -> None:
    global _PU_KEEP
    _PU_KEEP = pu_keep
    set_brl(threshold, mode)
    if threshold > 0:
        lwdetr.SetCriterion.loss_labels = _loss_labels_brl
        print(f"  [BRL] mode={mode} threshold={threshold}"
              + (f" pu_keep={pu_keep}" if mode == "pu" else ""))
    else:
        lwdetr.SetCriterion.loss_labels = _ORIGINAL
