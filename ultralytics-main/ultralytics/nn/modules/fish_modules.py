# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""FSDNet custom modules: StemS2D, WaveletBlock, RepC3k2, FishDetect(+Aux), FishDetectLSCD/LSDECD."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import C3k2, RepBottleneck
from .conv import Conv, RepConv
from .head import Detect


# ---------------------------------------------------------------------------
# StemS2D: detail-preserving downsample stem
# ---------------------------------------------------------------------------
class StemS2D(nn.Module):
    """Space-to-Depth stem that preserves fine-grained spatial details.

    Replaces the standard stride-2 convolution at the network entry with
    PixelUnshuffle (r=2) followed by RepConv channel projection, retaining
    local structure that stride convolutions discard.
    """

    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.s2d = nn.PixelUnshuffle(2)
        self.conv = RepConv(c1 * 4, c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.s2d(x))


# ---------------------------------------------------------------------------
# Haar DWT utility (fixed, non-learnable)
# ---------------------------------------------------------------------------
class _HaarDWT2D(nn.Module):
    """Fixed 2D Haar discrete wavelet transform."""

    def forward(self, x: torch.Tensor):
        # Pad to even spatial dims if needed
        _, _, h, w = x.shape
        if h % 2:
            x = F.pad(x, (0, 0, 0, 1), mode="reflect")
        if w % 2:
            x = F.pad(x, (0, 1, 0, 0), mode="reflect")

        x00 = x[..., 0::2, 0::2]
        x01 = x[..., 0::2, 1::2]
        x10 = x[..., 1::2, 0::2]
        x11 = x[..., 1::2, 1::2]

        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (x00 + x01 - x10 - x11) * 0.5
        hl = (x00 - x01 + x10 - x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        return ll, lh, hl, hh


# ---------------------------------------------------------------------------
# WaveletBlock: partial-channel frequency-domain enhancement
# ---------------------------------------------------------------------------
class WaveletBlock(nn.Module):
    """Partial-channel wavelet block for frequency-aware feature enhancement.

    Splits channels into a keep-path (identity) and a wavelet-path.  The
    wavelet path applies Haar DWT, refines LL/HF sub-bands with learnable
    convolutions, generates a low-frequency gate to suppress noisy high
    frequencies, then fuses and restores spatial resolution via upsampling.
    """

    def __init__(self, c1: int, c2: int, ratio: float = 0.5):
        super().__init__()
        assert c1 == c2, "WaveletBlock requires c1 == c2 (channel-preserving)"
        self.c_wav = int(c1 * ratio)
        self.c_keep = c1 - self.c_wav

        self.dwt = _HaarDWT2D()

        # LL (low-frequency) branch: refine + global gate
        self.ll_conv = Conv(self.c_wav, self.c_wav, 3)
        self.ll_pool = nn.AdaptiveAvgPool2d(1)
        hidden = max(self.c_wav // 4, 4)
        self.ll_gate = nn.Sequential(
            nn.Linear(self.c_wav, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, self.c_wav),
            nn.Sigmoid(),
        )

        # HF (high-frequency) branch: compress + refine
        self.hf_compress = Conv(self.c_wav * 3, self.c_wav, 1)
        self.hf_refine = Conv(self.c_wav, self.c_wav, 3)

        # Upsample-fuse
        self.fuse = Conv(self.c_wav * 2, self.c_wav, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.c_wav == 0:
            return x

        identity = x
        f_keep, f_wav = x.split([self.c_keep, self.c_wav], dim=1)
        th, tw = f_wav.shape[2], f_wav.shape[3]

        # DWT
        ll, lh, hl, hh = self.dwt(f_wav)

        # LL branch -> gate
        ll_feat = self.ll_conv(ll)
        gate = self.ll_pool(ll_feat).flatten(1)
        gate = self.ll_gate(gate).unsqueeze(-1).unsqueeze(-1)

        # HF branch -> gated
        hf = self.hf_compress(torch.cat([lh, hl, hh], dim=1))
        hf = self.hf_refine(hf)
        hf = gate * hf

        # Restore spatial resolution and fuse
        ll_up = F.interpolate(ll_feat, size=(th, tw), mode="bilinear", align_corners=False)
        hf_up = F.interpolate(hf, size=(th, tw), mode="bilinear", align_corners=False)
        f_wav_out = self.fuse(torch.cat([ll_up, hf_up], dim=1))

        out = torch.cat([f_keep, f_wav_out], dim=1)
        return out + identity


# ---------------------------------------------------------------------------
# RepC3k2: reparameterizable C3k2 block
# ---------------------------------------------------------------------------
class RepC3k2(C3k2):
    """C3k2 with internal bottlenecks replaced by RepBottleneck.

    Training uses multi-branch RepConv for stronger representation;
    inference can fold branches into a single conv for deployment.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, c3k: bool = False,
                 e: float = 0.5, attn: bool = False, g: int = 1, shortcut: bool = True):
        super().__init__(c1, c2, n, c3k=False, e=e, attn=False, g=g, shortcut=shortcut)
        self.m = nn.ModuleList(RepBottleneck(self.c, self.c, shortcut, g) for _ in range(n))


# ---------------------------------------------------------------------------
# FishDetect: shape-aware detection head
# ---------------------------------------------------------------------------
class FishDetect(Detect):
    """Detection head with auxiliary fish-shape endpoint prediction.

    Extends the standard Detect head with a lightweight keypoint branch
    that predicts 4 normalised offsets (Δx_h, Δy_h, Δx_t, Δy_t) for fish
    head/tail endpoints relative to the box centre.

    During training the branch feeds kpt_loss and shape_loss.  During
    inference an optional shape-gate multiplies detection scores by a
    geometric confidence derived from the predicted axis, suppressing
    non-fish false positives (disabled by default, toggle via shape_gate).
    """

    shape_gate = False  # experimental inference-time shape gating
    nk = 4  # 2 keypoints × 2 dimensions

    def __init__(self, nc: int = 80, reg_max: int = 16, end2end: bool = False, ch: tuple = ()):
        super().__init__(nc, reg_max, end2end, ch)
        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch
        )
        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    # -- head property overrides -------------------------------------------
    @property
    def one2many(self):
        return dict(box_head=self.cv2, cls_head=self.cv3, kpt_head=self.cv4)

    @property
    def one2one(self):
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, kpt_head=self.one2one_cv4)

    # -- forward ------------------------------------------------------------
    def forward_head(self, x: list[torch.Tensor], box_head=None, cls_head=None, kpt_head=None):
        preds = super().forward_head(x, box_head, cls_head)
        if kpt_head is not None and preds:
            bs = x[0].shape[0]
            preds["kpts"] = torch.cat(
                [kpt_head[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], 2
            )
        return preds

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        dbox = self._get_decode_boxes(x)
        scores = x["scores"].sigmoid()
        if self.shape_gate and "kpts" in x:
            scores = scores * self._shape_confidence(x["kpts"])
        return torch.cat((dbox, scores), 1)

    def _shape_confidence(self, kpts: torch.Tensor) -> torch.Tensor:
        """Geometric confidence: longer predicted head-tail axis → higher score."""
        dx = kpts[:, 2:3] - kpts[:, 0:1]
        dy = kpts[:, 3:4] - kpts[:, 1:2]
        axis_len = (dx ** 2 + dy ** 2 + 1e-8).sqrt()
        return (axis_len * 5.0 - 1.5).sigmoid()

    # -- fuse / bias ---------------------------------------------------------
    def fuse(self) -> None:
        super().fuse()
        self.cv4 = None

    def bias_init(self):
        super().bias_init()
        for conv_seq in self.cv4:
            conv_seq[-1].bias.data.zero_()
        if self.end2end:
            for conv_seq in self.one2one_cv4:
                conv_seq[-1].bias.data.zero_()


class FishDetectAux(FishDetect):
    """FishDetect with an extra train-time auxiliary detection branch.

    Main branch remains identical to FishDetect (box/cls + kpt). In training,
    a lightweight auxiliary box/cls branch on the same feature pyramid emits:
      - aux_boxes: [B, 4*reg_max, N]
      - aux_scores: [B, nc, N]
    and is consumed by FishDetectionLoss as deep supervision.

    In inference/export, aux outputs are ignored so deployment I/O is unchanged.
    """

    def __init__(self, nc: int = 80, reg_max: int = 16, end2end: bool = False, ch: tuple = ()):
        super().__init__(nc=nc, reg_max=reg_max, end2end=end2end, ch=ch)
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))
        self.cv2_aux = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3_aux = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(RepConv(x, x), Conv(x, c3, 1)),
                nn.Sequential(RepConv(c3, c3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )

    def _forward_aux(self, x: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        bs = x[0].shape[0]
        aux_boxes = torch.cat([self.cv2_aux[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        aux_scores = torch.cat([self.cv3_aux[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
        return dict(aux_boxes=aux_boxes, aux_scores=aux_scores)

    def forward(self, x):
        preds = self.forward_head(x, **self.one2many)
        if self.end2end:
            x_detach = [xi.detach() for xi in x]
            one2one = self.forward_head(x_detach, **self.one2one)
            if self.training:
                one2many = preds
                one2many.update(self._forward_aux(x))
                preds = {"one2many": one2many, "one2one": one2one}
            else:
                preds = {"one2many": preds, "one2one": one2one}
        elif self.training:
            preds.update(self._forward_aux(x))
        if self.training:
            return preds
        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def bias_init(self):
        super().bias_init()
        for i, (a, b) in enumerate(zip(self.cv2_aux, self.cv3_aux)):
            a[-1].bias.data[:] = 2.0
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)

    def fuse(self) -> None:
        super().fuse()
        self.cv2_aux = None
        self.cv3_aux = None


# ---------------------------------------------------------------------------
# DualGateConcat: bilateral-reweighted Concat replacement for the FPN neck
# ---------------------------------------------------------------------------
class _HSigmoid(nn.Module):
    """Deployment-friendly hard-sigmoid: ReLU6(x + 3) / 6."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu6(x + 3.0, inplace=True) / 6.0


class _ConvBNReLU(nn.Module):
    """1x1 / 3x3 Conv + BN + ReLU (ReLU chosen for quantisation friendliness)."""

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, g: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, padding=k // 2, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DualGateConcat(nn.Module):
    """Gated bilateral re-weighting before Concat (drop-in Concat replacement).

    Given two same-scale inputs ``[x_up, x_lat]`` (e.g. upsampled deep
    feature and lateral shallow feature in an FPN), this module:

      1. Projects each branch with a 1x1 Conv-BN-ReLU to a shared hidden dim.
      2. Concats the two projections along channels.
      3. Applies a depthwise 3x3 conv followed by a 1x1 pointwise conv to
         produce a joint gating tensor with ``c_up + c_lat`` channels.
      4. Splits the gate into ``(g_up, g_lat)``.
      5. Hard-sigmoid -> weight maps in ``[0, 1]``.
      6. Residual re-scaling: ``x' = x * (1 + gamma * w)`` with per-channel
         learnable ``gamma`` initialised to zero (identity at init).
      7. Returns ``cat([x_up', x_lat'], dim=1)`` so output channel count
         equals the original ``Concat`` (c_up + c_lat); downstream layers
         remain unchanged.

    Input channels of the two branches may differ; the 1x1 alignment is
    done inside the module when building the gate.
    """

    def __init__(
        self,
        c_up: int,
        c_lat: int | None = None,
        gamma_init: float = 0.0,
        hidden: int | None = None,
    ):
        super().__init__()
        c_lat = c_up if c_lat is None else c_lat
        self.c_up = c_up
        self.c_lat = c_lat
        # Shared hidden dim for gate computation; conservative to stay light.
        c_h = hidden if hidden is not None else max(min(c_up, c_lat), 8)
        self.c_h = c_h

        # Step 1: lightweight projections (gate-only, do NOT change main-path channels).
        self.proj_up = _ConvBNReLU(c_up, c_h, k=1)
        self.proj_lat = _ConvBNReLU(c_lat, c_h, k=1)

        # Step 3: joint gating network  (DW 3x3 -> PW 1x1).
        self.gate_dw = _ConvBNReLU(2 * c_h, 2 * c_h, k=3, g=2 * c_h)
        self.gate_pw = nn.Conv2d(2 * c_h, c_up + c_lat, kernel_size=1, bias=True)

        # Step 5: deployment-friendly hard-sigmoid.
        self.hsigmoid = _HSigmoid()

        # Step 6: per-channel learnable scales.
        #   gamma_init=0.0 -> exact identity at init (safest).
        #   gamma_init>0   -> module is "pre-warmed", gates contribute from epoch 0.
        #                     Recommend 0.05 ~ 0.1 if single fusion point or strong baseline.
        self.gamma_init = float(gamma_init)
        self.gamma_up = nn.Parameter(torch.full((1, c_up, 1, 1), self.gamma_init))
        self.gamma_lat = nn.Parameter(torch.full((1, c_lat, 1, 1), self.gamma_init))

    def forward(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 2, (
            f"DualGateConcat expects [x_up, x_lat], got {type(x).__name__} of length "
            f"{len(x) if hasattr(x, '__len__') else 'N/A'}"
        )
        x_up, x_lat = x

        p_up = self.proj_up(x_up)
        p_lat = self.proj_lat(x_lat)

        joint = torch.cat([p_up, p_lat], dim=1)
        gate = self.gate_pw(self.gate_dw(joint))

        g_up, g_lat = gate.split([self.c_up, self.c_lat], dim=1)
        w_up = self.hsigmoid(g_up)
        w_lat = self.hsigmoid(g_lat)

        x_up_cal = x_up * (1.0 + self.gamma_up * w_up)
        x_lat_cal = x_lat * (1.0 + self.gamma_lat * w_lat)

        return torch.cat([x_up_cal, x_lat_cal], dim=1)


# ---------------------------------------------------------------------------
# FishDetectLSCD / FishDetectLSDECD
#   Lightweight Shared (Detail-Enhanced) Convolutional Detection heads that
#   preserve FishDetect's kpt (+ shape) auxiliary supervision.
#
#   Design (参考 LSCD / LSDECD 的 "共享干 + 轻量头" 思路)：
#     per-scale 1x1 stem  →  shared Conv3x3 × N  →  per-scale 1x1 heads
#     heads: cv2 (box / DFL)、cv3 (cls)、cv4 (kpt)
#
#   LSCD : shared 用普通 Conv3x3
#   LSDECD: shared 用 RepConv（训练多支路 3x3 + 1x1 + id-BN，导出前
#           RepConv.fuse_convs() 合并为单 3x3，"detail-enhanced" 表达更强）
#
#   注意事项：
#     - 共享段的 BN 在三个尺度之间统计量是“平均”的，是 LSCD 一族已知的权衡；
#       per-scale 差异靠前面 1x1 stem 的独立 BN + 后面 1x1 head 的 bias 解耦。
#     - kpt 头 (cv4) 也挂在共享特征上，训练时 FishDetectionLoss 照旧生效；
#       推理默认不走 kpt（shape_gate 关），与原 FishDetect 一致。
#     - kpt/shape 的 epoch 与 warm-up 在 ``ultralytics.utils.fish_loss.FishDetectionLoss`` 中
#       统一处理：非 E2E 训练依赖 ``FishDetectionLoss.update()`` 自增周期；E2E 时由
#       ``FishE2ELoss.update()`` 对内层 ``FishDetectionLoss`` 调 ``set_epoch``。
#       LSCD / LSDECD 与 ``FishDetect`` 共用该 loss，无独立第二份实现、也无单独“再修一次”的模块。
# ---------------------------------------------------------------------------
class FishDetectLSCD(FishDetect):
    """Lightweight Shared Conv Detection head + Fish kpt branch.

    Replaces the standard per-scale cv2/cv3/cv4 stacks with:
      - per-scale 1x1 projection (stem) to a shared channel ``c_shared``
      - a shared ``Conv3x3 × shared_stack`` block applied to each scale
      - minimal per-scale 1x1 heads for box / cls / kpt
    """

    # Number of shared Conv3x3 (or RepConv) stacked in the shared block.
    shared_stack: int = 2

    def __init__(self, nc: int = 1, reg_max: int = 16, end2end: bool = False, ch: tuple = ()):
        # Parent builds full cv2/cv3/cv4 + dfl + (one2one_*) from ch;
        # we rebuild cv2/cv3/cv4 as lightweight 1x1 heads taking c_shared.
        super().__init__(nc=nc, reg_max=reg_max, end2end=end2end, ch=ch)
        c_shared = self._default_c_shared(ch)
        self.c_shared = c_shared

        # per-scale 1x1 stem（每尺度独立 BN，负责把 backbone 通道统一到 c_shared）
        self.stem = nn.ModuleList(Conv(c, c_shared, 1) for c in ch)

        # shared conv block（所有尺度共享同一份权重）
        self.shared = self._build_shared(c_shared)

        # 轻量 per-scale 1x1 heads；包成 Sequential 以兼容 Detect.bias_init 的 [-1] 访问
        self.cv2 = nn.ModuleList(
            nn.Sequential(nn.Conv2d(c_shared, 4 * self.reg_max, 1)) for _ in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(nn.Conv2d(c_shared, self.nc, 1)) for _ in ch
        )
        self.cv4 = nn.ModuleList(
            nn.Sequential(nn.Conv2d(c_shared, self.nk, 1)) for _ in ch
        )

        if end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    # -- hyperparams / sub-blocks ------------------------------------------
    @staticmethod
    def _default_c_shared(ch: tuple) -> int:
        """Pick a reasonable shared-channel width (>= 64, typically min(ch))."""
        return max(min(ch), 64)

    def _build_shared(self, c: int) -> nn.Sequential:
        return nn.Sequential(*[Conv(c, c, 3) for _ in range(self.shared_stack)])

    # -- forward ------------------------------------------------------------
    def _shared_features(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        return [self.shared(self.stem[i](x[i])) for i in range(self.nl)]

    def forward(self, x):
        # Pre-compute shared features for all scales, then reuse
        # FishDetect/Detect.forward for the rest (one2many / one2one / kpt / export).
        return super().forward(self._shared_features(x))


class FishDetectLSDECD(FishDetectLSCD):
    """LSCD variant where the shared 3x3 stack uses ``RepConv``.

    Training: 3x3 + 1x1 + identity-BN 多支路 (RepConv)；
    Export  : RepConv.fuse_convs() 合并为单个 3x3 Conv，等效结构更轻。
    """

    def _build_shared(self, c: int) -> nn.Sequential:
        return nn.Sequential(*[RepConv(c, c) for _ in range(self.shared_stack)])
