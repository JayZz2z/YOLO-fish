# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""FSDNet losses: keypoint regression + shape consistency on top of standard detection loss."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .loss import E2ELoss, v8DetectionLoss


class FishDetectionLoss(v8DetectionLoss):
    """Detection loss extended with fish-shape keypoint and shape-consistency terms.

    Produces a 5-element loss vector: [box, cls, dfl, kpt, shape].
    The kpt and shape terms are only active when ``preds`` contains a
    ``"kpts"`` key (i.e. when using FishDetect head).

    Auxiliary losses are gradually introduced via linear warmup to avoid
    noisy pseudo-label gradients from disrupting early box/cls convergence.
    """

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):
        super().__init__(model, tal_topk, tal_topk2)
        self.lambda_kpt = getattr(self.hyp, "kpt", 0.1)
        self.lambda_shape = getattr(self.hyp, "shape", 0.02)
        self.lambda_aux_det = getattr(self.hyp, "aux_det", 0.2)
        self.shape_margin = getattr(self.hyp, "shape_margin", 0.15)
        self.aux_warmup = getattr(self.hyp, "aux_warmup", 30)
        self.total_epochs = getattr(self.hyp, "epochs", 900)
        self._cur_epoch = 0

    def set_epoch(self, epoch: int):
        self._cur_epoch = epoch

    def update(self) -> None:
        """Match DetectionTrainer: called once at end of each training epoch (if method exists)."""
        self._cur_epoch += 1

    def _aux_weight(self) -> float:
        """Linear warmup from 0 to 1 over aux_warmup epochs."""
        if self.aux_warmup <= 0:
            return 1.0
        return min(max(self._cur_epoch - 10, 0) / self.aux_warmup, 1.0)

    # --------------------------------------------------------------------- #
    # Core override
    # --------------------------------------------------------------------- #
    def get_assigned_targets_and_loss(
        self, preds: dict[str, torch.Tensor], batch: dict[str, Any]
    ) -> tuple:
        """Standard detection loss + kpt/shape auxiliary losses."""
        assigned, loss_det, _ = super().get_assigned_targets_and_loss(preds, batch)
        fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor = assigned

        loss = torch.zeros(5, device=self.device)
        loss[:3] = loss_det

        w = self._aux_weight()
        if w > 0 and "kpts" in preds and fg_mask.sum() > 0:
            kpt_loss, shape_loss = self._fish_losses(
                preds["kpts"], fg_mask, target_bboxes
            )
            loss[3] = kpt_loss * self.lambda_kpt * w
            loss[4] = shape_loss * self.lambda_shape * w

        # Optional train-time auxiliary detection branch (FishDetectAux):
        # apply the same assigner/loss as the main detection branch and add
        # it to box/cls/dfl with a small gain and warmup.
        if (
            w > 0
            and self.lambda_aux_det > 0
            and "aux_boxes" in preds
            and "aux_scores" in preds
            and "feats" in preds
        ):
            aux_preds = {
                "boxes": preds["aux_boxes"],
                "scores": preds["aux_scores"],
                "feats": preds["feats"],
            }
            _, aux_det_loss, _ = super().get_assigned_targets_and_loss(aux_preds, batch)
            loss[:3] += aux_det_loss * self.lambda_aux_det * w

        return assigned, loss, loss.detach()

    # --------------------------------------------------------------------- #
    # Fish-specific loss helpers
    # --------------------------------------------------------------------- #
    def _fish_losses(
        self,
        pred_kpts: torch.Tensor,
        fg_mask: torch.Tensor,
        target_bboxes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute keypoint regression and shape consistency losses.

        Args:
            pred_kpts: [B, 4, N] raw kpt predictions (dx_h, dy_h, dx_t, dy_t).
            fg_mask: [B, N] boolean foreground mask.
            target_bboxes: [B, N, 4] assigned GT boxes in xyxy pixel coords.

        Returns:
            kpt_loss, shape_loss (scalars).
        """
        pred = pred_kpts.permute(0, 2, 1)  # [B, N, 4]
        fg_pred = pred[fg_mask]  # [Nfg, 4]
        fg_boxes = target_bboxes[fg_mask]  # [Nfg, 4] xyxy

        gt_kpts = self._pseudo_kpts(fg_boxes)  # [Nfg, 4]
        kpt_loss = self._symmetric_l1(fg_pred, gt_kpts)

        shape_loss = self._shape_consistency(fg_pred, fg_boxes)
        return kpt_loss, shape_loss

    # ---- pseudo keypoint generation -------------------------------------- #
    @staticmethod
    def _pseudo_kpts(boxes: torch.Tensor) -> torch.Tensor:
        """Generate pseudo head/tail offsets from bounding boxes.

        For elongated objects the endpoints lie on the longer axis:
          horizontal fish  →  (-0.5, 0, 0.5, 0)
          vertical   fish  →  (0, -0.5, 0, 0.5)

        Returns: [Nfg, 4] normalised offsets.
        """
        w = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-6)
        h = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-6)
        horiz = (w >= h).float()
        return torch.stack(
            [
                -0.5 * horiz,
                -0.5 * (1.0 - horiz),
                0.5 * horiz,
                0.5 * (1.0 - horiz),
            ],
            dim=-1,
        )

    # ---- symmetric (order-agnostic) L1 ---------------------------------- #
    @staticmethod
    def _symmetric_l1(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """L_kpt = min(||p_h-g_h||+||p_t-g_t||, ||p_h-g_t||+||p_t-g_h||)."""
        d1 = F.l1_loss(pred[:, :2], gt[:, :2], reduction="none").sum(-1) + F.l1_loss(
            pred[:, 2:], gt[:, 2:], reduction="none"
        ).sum(-1)
        d2 = F.l1_loss(pred[:, :2], gt[:, 2:], reduction="none").sum(-1) + F.l1_loss(
            pred[:, 2:], gt[:, :2], reduction="none"
        ).sum(-1)
        return torch.minimum(d1, d2).mean()

    # ---- shape consistency ----------------------------------------------- #
    def _shape_consistency(
        self, pred: torch.Tensor, boxes: torch.Tensor
    ) -> torch.Tensor:
        """Penalise predicted axis-length ratio outside a plausible band.

        r_pred = predicted_axis_length / box_diagonal
        r_gt   = max(w, h) / diagonal  (reference ratio from GT box)
        Loss fires when |r_pred − r_gt| > margin.
        """
        w = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-6)
        h = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-6)

        dx = (pred[:, 2] - pred[:, 0]) * w
        dy = (pred[:, 3] - pred[:, 1]) * h
        axis_len = (dx ** 2 + dy ** 2 + 1e-8).sqrt()
        diag = (w ** 2 + h ** 2 + 1e-8).sqrt()

        r_pred = axis_len / diag
        r_gt = torch.max(w, h) / diag

        return torch.clamp(torch.abs(r_pred - r_gt) - self.shape_margin, min=0.0).mean()


# --------------------------------------------------------------------------- #
# E2E wrapper
# --------------------------------------------------------------------------- #
class FishE2ELoss(E2ELoss):
    """End-to-end loss that uses FishDetectionLoss for both o2m and o2o branches."""

    def __init__(self, model):
        super().__init__(model, loss_fn=FishDetectionLoss)

    def update(self) -> None:
        super().update()
        self.one2many.set_epoch(self.updates)
        self.one2one.set_epoch(self.updates)
