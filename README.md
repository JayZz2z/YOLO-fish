# YOLO-fish

基于 [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) 的鱼类检测训练仓库：在自带 `ultralytics-main` 中扩展 **FishDetect**、**RepC3k2**、**DualGateConcat（BRM）** 等模块，模型结构见 `ultralytics-main/ultralytics/cfg/models/v8/YOLO_fish.yaml`。默认训练脚本开启 **APT-TAL** 及 box/dfl 折中系数，参数写在 `train_fsdnet_v8.py` 内，无需命令行传参。

## 环境

- Python 3.10+（开发机曾用 3.13）
- 建议使用 NVIDIA GPU + 对应 CUDA 的 [PyTorch](https://pytorch.org)

```bash
cd YOLO-fish
pip install -r requirements.txt
```

`requirements.txt` 中的 `torch` / `torchvision` 可按本机 CUDA 从官网安装，不必与文件里版本完全一致。

本仓库通过 `train_fsdnet_v8.py` 把 `ultralytics-main` 加入 `sys.path`，**不强制** `pip install ultralytics`。若希望全局 `import ultralytics`，可执行：

```bash
pip install -e ./ultralytics-main
```

## 数据

按 Ultralytics 格式准备数据集，并在仓库根目录放置：

```text
dataset/
  data.yaml    # 训练/验证路径与类别名
  ...
```

`dataset/` 默认被 `.gitignore` 忽略，需自行准备。`data.yaml` 中的路径建议写成相对仓库根目录的形式，便于迁移。

## 预训练权重

根目录提供 **`yolov8n.pt`**（与脚本中 `DEFAULT_WEIGHTS` 一致）。若缺失，可从 Ultralytics 官方渠道下载同名文件放到仓库根目录。

## 训练

确认 `dataset/data.yaml` 与 `yolov8n.pt` 就绪后：

```bash
python train_fsdnet_v8.py
```

- **模型**：`ultralytics-main/ultralytics/cfg/models/v8/YOLO_fish.yaml`
- **输出目录**：由脚本内 `TRAIN_OVERRIDES` 指定（当前为 `output/`，实验名为 `YOLO_fish_e900`）
- 修改 epoch、batch、设备、APT-TAL 等：直接编辑 `train_fsdnet_v8.py` 中的常量或 `TRAIN_OVERRIDES` 字典

## 仓库结构（简要）

| 路径 | 说明 |
|------|------|
| `train_fsdnet_v8.py` | 单次训练入口 |
| `ultralytics-main/` | 内嵌 Ultralytics 源码及 Fish 相关改动 |
| `ultralytics-main/ultralytics/cfg/models/v8/YOLO_fish.yaml` | 模型结构（Rep + FishDetect + DualGateConcat） |
| `requirements.txt` | Python 依赖 |
| `output/YOLO_fish_e900/` | 示例训练产物（曲线、日志、`weights/` 等） |

## 许可证

`ultralytics-main` 沿用 Ultralytics 原许可证（见 `ultralytics-main/LICENSE`）。使用本仓库时请一并遵守上游协议。

## 链接

- 远程仓库：<https://github.com/JayZz2z/YOLO-fish>
