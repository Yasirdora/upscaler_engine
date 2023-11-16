"""Importable model loading, shared by convert.py and the reference oracle.

`convert.py` and `tools/reference/run_reference.py` must reconstruct the *same* network from the
same checkpoint; two copies of an architecture that both load `strict=True` can still differ in
their forward pass, and the oracle would then certify a model nobody converted. One definition,
imported by both, removes that failure mode.

Attribution: the RRDBNet architecture is reimplemented from the public description in the
Real-ESRGAN project (BSD-3-Clause, https://github.com/xinntao/Real-ESRGAN); module and parameter
names follow upstream so the published checkpoint loads with strict=True. No code from Upscayl
(AGPL-3.0) or any GPL/AGPL source was consulted or copied.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelLoadError(RuntimeError):
    """The checkpoint or architecture config cannot produce a usable model."""


def log(message: str) -> None:
    print(message, flush=True)


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat: int, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    """RRDBNet at ×4 or ×2.

    The ×2 variant is not a different network: it is the same 23-block body fed a
    `pixel_unshuffle`d input. Unshuffling by 2 quarters the spatial positions the body sees and
    quadruples its input channels, so `conv_first` becomes 12→64 rather than 3→64 — the whole
    parameter difference, 5,184 weights. The body still upsamples ×4 internally; the ×2 net result
    is the unshuffle cancelling half of it.

    That is also why ×2 is cheap: the expensive part runs at half resolution in each dimension.
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
        scale: int = 4,
    ) -> None:
        super().__init__()
        if scale not in (2, 4):
            raise ModelLoadError(f"unsupported RRDBNet scale {scale} (expected 2 or 4)")
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = F.pixel_unshuffle(x, downscale_factor=2) if self.scale == 2 else x
        feat = self.conv_first(feat)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2.0, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2.0, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


class SRVGGNetCompact(nn.Module):
    """The compact VGG-style generator `realesr-general-x4v3` ships.

    Added because the x4plus graph misses certification spec §7.2's A14 threshold by 3.5x, and
    profiling established the cause is not schedulable — the ANE is saturated at 1026/1026 ops
    resident, batching gives negative relief, and larger tile shapes reduce no work. Only a cheaper
    graph closes it, which is the lever §7.2 reserved: "a miss means switch models, not degrade
    output."

    Plain sequential convolutions with a nearest-neighbour residual, not RRDBNet's dense blocks —
    x4v3 measures 1,213,296 parameters against x4plus's 16,697,987, and ~14.8x fewer MACs at 512².

    Attribution, on the same terms as `RRDBNet` above: reimplemented from the public description in
    the Real-ESRGAN project (BSD-3-Clause, https://github.com/xinntao/Real-ESRGAN); module and
    parameter names follow upstream so the published checkpoint loads with `strict=True`. No code
    from Upscayl (AGPL-3.0) or any GPL/AGPL source was consulted or copied.
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_conv: int = 32,
        upscale: int = 4,
        act_type: str = "prelu",
    ) -> None:
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(self._activation(act_type, num_feat))
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(self._activation(act_type, num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    @staticmethod
    def _activation(act_type: str, num_feat: int) -> nn.Module:
        if act_type == "prelu":
            return nn.PReLU(num_parameters=num_feat)
        if act_type == "relu":
            return nn.ReLU(inplace=True)
        if act_type == "leakyrelu":
            return nn.LeakyReLU(negative_slope=0.1, inplace=True)
        raise ModelLoadError(f"unsupported activation '{act_type}' (expected prelu|relu|leakyrelu)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        # The network learns a residual over a nearest-neighbour upsample, so the base must be
        # added back. Omitting it loads strict=True and still produces a plausible-looking but
        # wrong image — which is exactly the failure one shared definition exists to prevent.
        return out + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


def build_model(config: dict, weights_path: Path) -> tuple[nn.Module, int]:
    """Reconstruct the configured architecture from its checkpoint.

    Dispatches on `architecture.type`. Both branches load `strict=True` so architecture drift
    fails loudly rather than producing a network that runs and certifies the wrong thing.
    """
    arch = config["architecture"]
    kind = arch["type"]
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    key = config["weights"]["state_dict_key"]
    state_dict = checkpoint[key] if isinstance(checkpoint, dict) and key in checkpoint else checkpoint
    model: nn.Module
    if kind == "RRDBNet":
        model = RRDBNet(
            num_in_ch=arch["num_in_ch"],
            num_out_ch=arch["num_out_ch"],
            num_feat=arch["num_feat"],
            num_block=arch["num_block"],
            num_grow_ch=arch["num_grow_ch"],
            scale=arch.get("scale", 4),
        )
    elif kind == "SRVGGNetCompact":
        model = SRVGGNetCompact(
            num_in_ch=arch["num_in_ch"],
            num_out_ch=arch["num_out_ch"],
            num_feat=arch["num_feat"],
            num_conv=arch["num_conv"],
            upscale=arch["upscale"],
            act_type=arch.get("act_type", "prelu"),
        )
    else:
        raise ModelLoadError(
            f"unsupported architecture '{kind}' (expected RRDBNet or SRVGGNetCompact)"
        )
    model.load_state_dict(state_dict, strict=True)  # strict: architecture drift fails loudly
    model.eval()
    count = sum(p.numel() for p in model.parameters())
    log(f"[model] loaded strict=True from '{key}', {count:,} parameters")
    return model, count
