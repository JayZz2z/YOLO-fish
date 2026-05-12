"""单次训练：YOLO_fish + APT-TAL「温和 + 拉定位」（mAP50 / mAP50-95 折中，需自测）。

结构：`ultralytics/cfg/models/v8/YOLO_fish.yaml`（Rep + FishDetect + neck 全 DualGateConcat）。
运行：``python train_fsdnet_v8.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "ultralytics-main"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.models.yolo.detect.fish_train import FishDetectionTrainer  # noqa: E402
from ultralytics.utils import LOGGER  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent
CFG_DIR = ROOT / "ultralytics" / "cfg" / "models" / "v8"

# 模型配置位于 ultralytics-main/ultralytics/cfg/models/v8/YOLO_fish.yaml（经 CFG_DIR 拼接）
MODEL_YAML = "YOLO_fish.yaml"
RUN_NAME = "YOLO_fish_e900"
APT_TAL = True
APT_TAL_POWER = 1.3
APT_TAL_QUALITY_GAMMA = 0.9
BOX = 8.0
DFL = 1.65

DEFAULT_WEIGHTS: Path | str | None = REPO_ROOT / "yolov8n.pt"

TRAIN_OVERRIDES: dict = {
    "data": str(REPO_ROOT / "dataset" / "data.yaml"),
    "epochs": 900,
    "imgsz": 320,
    "batch": 32,
    "device": "0",
    "workers": 4,
    "project": str(REPO_ROOT / "output"),
    "exist_ok": False,
    "kpt": 0.1,
    "shape": 0.02,
    "patience": 100,
    "model": str(CFG_DIR / MODEL_YAML),
    "name": RUN_NAME,
    "apt_tal": APT_TAL,
    "apt_tal_power": APT_TAL_POWER,
    "apt_tal_quality_gamma": APT_TAL_QUALITY_GAMMA,
    "box": BOX,
    "dfl": DFL,
    "pretrained": str(DEFAULT_WEIGHTS) if DEFAULT_WEIGHTS else False,
}


def main() -> None:
    overrides = dict(TRAIN_OVERRIDES)
    trainer = FishDetectionTrainer(overrides=overrides)
    pre = overrides.get("pretrained", True)
    init_msg = "随机初值 (pretrained=False)" if pre is False else f"预训练/权重: {pre}"
    LOGGER.info(
        f"YOLOv8n 单次训练 [YOLO_fish + APT-TAL tuned] 输出: {trainer.save_dir}\n"
        f"  yaml={overrides['model']}\n  {init_msg}"
    )
    trainer.train()


if __name__ == "__main__":
    main()
