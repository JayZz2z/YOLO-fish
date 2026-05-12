# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""FSDNet trainer and model wrappers."""

from __future__ import annotations

from copy import copy
from typing import Any

from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.fish_loss import FishDetectionLoss, FishE2ELoss

from .train import DetectionTrainer


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class FSDNetModel(DetectionModel):
    """DetectionModel subclass that wires FishDetect-aware loss functions."""

    def init_criterion(self):
        return FishE2ELoss(self) if self.end2end else FishDetectionLoss(self)


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #
class FishDetectionTrainer(DetectionTrainer):
    """Trainer for FSDNet with extended 5-component loss logging.

    Extends DetectionTrainer to:
    * build an FSDNetModel (which uses FishDetect head + fish losses),
    * log five loss terms: box_loss, cls_loss, dfl_loss, kpt_loss, shape_loss.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        model = FSDNetModel(cfg, nc=self.data["nc"], ch=self.data.get("channels", 3), verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = ("box_loss", "cls_loss", "dfl_loss", "kpt_loss", "shape_loss")
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
