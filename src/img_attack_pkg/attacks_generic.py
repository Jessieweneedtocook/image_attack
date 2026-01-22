# attacks_generic.py
# Goal:
# Accepts an input tensor in common layouts (CHW / HWC / BCHW / BHWC / HW)
# Internally converts it to BCHW in the [-1, 1] range
# Applies the same attacks using the original function names and parameters (Geometric + Signal)
# Returns the output in the same layout, dtype, and (as much as possible) the same value range as the input

import io
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from PIL import Image
import torchvision.io as tvio
try:
    import kornia
    import kornia.filters as Kf
    import kornia.enhance as Ke
    _HAS_KORNIA = True
except Exception:
    _HAS_KORNIA = False




# ============================================================
# 1) Shape
# ============================================================
def _detect_layout(x: torch.Tensor) -> str:
    """
    يحدد layout للمدخل:
    - HW
    - CHW
    - HWC
    - BCHW
    - BHWC
    """
    if x.dim() == 2:
        return "HW"
    if x.dim() == 3:
        # CHW vs HWC
        return "CHW" if x.shape[0] <= 4 else "HWC"
    if x.dim() == 4:
        # BCHW vs BHWC
        return "BCHW" if x.shape[1] <= 4 else "BHWC"
    raise ValueError(f"Unsupported tensor dim={x.dim()} with shape={tuple(x.shape)}")


def _to_bchw(x: torch.Tensor):
    """
    يحول x إلى BCHW ويرجع:
    - x_bchw
    - meta dict لاسترجاع الشكل الأصلي
    """
    layout = _detect_layout(x)
    meta = {
        "layout": layout,
        "dtype": x.dtype,
        "device": x.device,
    }

    if layout == "HW":
        x_bchw = x.unsqueeze(0).unsqueeze(0)               # 1x1xHxW
    elif layout == "CHW":
        x_bchw = x.unsqueeze(0)                            # 1xCxHxW
    elif layout == "HWC":
        x_bchw = x.permute(2, 0, 1).unsqueeze(0)          # 1xCxHxW
    elif layout == "BCHW":
        x_bchw = x
    elif layout == "BHWC":
        x_bchw = x.permute(0, 3, 1, 2)                    # BxCxHxW
    else:
        raise ValueError(f"Unsupported layout={layout}")

    return x_bchw, meta


def _from_bchw(x_bchw: torch.Tensor, meta: dict) -> torch.Tensor:
    """يرجع BCHW إلى نفس layout الأصلي."""
    layout = meta["layout"]

    if layout == "HW":
        return x_bchw[0, 0]                                # HW
    if layout == "CHW":
        return x_bchw[0]                                   # CHW
    if layout == "HWC":
        return x_bchw[0].permute(1, 2, 0)                  # HWC
    if layout == "BCHW":
        return x_bchw
    if layout == "BHWC":
        return x_bchw.permute(0, 2, 3, 1)                  # BHWC

    raise ValueError(f"Unsupported layout={layout}")


def _detect_range_mode(x: torch.Tensor) -> str:
    """
    يحدد مجال/نوع قيم المدخل (Heuristic):
    - "uint8_255"
    - "float_0_1"
    - "float_0_255"
    - "float_-1_1"
    """
    if not torch.is_floating_point(x):
        return "uint8_255"

    safe = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x_min = float(safe.min().item())
    x_max = float(safe.max().item())

    # داخل [-1,1] (قد تكون 0..1 أو -1..1)
    if x_min >= -1.0001 and x_max <= 1.0001:
        return "float_0_1" if x_min >= -0.0001 else "float_-1_1"

    # داخل 0..255 تقريبًا
    if x_min >= -0.0001 and x_max <= 255.0001:
        return "float_0_1" if x_max <= 1.0001 else "float_0_255"

    # fallback
    return "float_-1_1"


def _to_minus1_1(x: torch.Tensor, mode: str) -> torch.Tensor:
    """يحـوّل x إلى float ضمن [-1,1]."""
    if mode == "uint8_255":
        x01 = (x.to(torch.float32) / 255.0).clamp(0.0, 1.0)
        return (x01 * 2.0 - 1.0).clamp(-1.0, 1.0)

    if mode == "float_0_1":
        x01 = x.to(torch.float32).clamp(0.0, 1.0)
        return (x01 * 2.0 - 1.0).clamp(-1.0, 1.0)

    if mode == "float_0_255":
        x01 = (x.to(torch.float32) / 255.0).clamp(0.0, 1.0)
        return (x01 * 2.0 - 1.0).clamp(-1.0, 1.0)

    # float_-1_1
    return x.to(torch.float32).clamp(-1.0, 1.0)


def _from_minus1_1(x_m11: torch.Tensor, mode: str, orig_dtype: torch.dtype) -> torch.Tensor:
    """يرجع من [-1,1] إلى نفس المجال/الداتا تايب الأصلي."""
    x_m11 = x_m11.clamp(-1.0, 1.0)

    if mode == "uint8_255":
        x01 = (x_m11 + 1.0) / 2.0
        x255 = (x01 * 255.0).round().clamp(0.0, 255.0)
        return x255.to(torch.uint8)

    if mode == "float_0_1":
        x01 = (x_m11 + 1.0) / 2.0
        return x01.clamp(0.0, 1.0).to(orig_dtype)

    if mode == "float_0_255":
        x01 = (x_m11 + 1.0) / 2.0
        x255 = (x01 * 255.0).clamp(0.0, 255.0)
        return x255.to(orig_dtype)

    # float_-1_1
    return x_m11.to(orig_dtype)


def _apply_attack_preserve(x: torch.Tensor, attack_fn, *args, **kwargs) -> torch.Tensor:
    """
    Wrapper:
    - يحول الشكل إلى BCHW
    - يحول المجال إلى [-1,1]
    - يطبق الهجوم على BCHW
    - يرجع لنفس الشكل + dtype + المجال الأصلي
    """
    x_bchw, meta = _to_bchw(x)
    range_mode = _detect_range_mode(x_bchw)

    x_m11 = _to_minus1_1(x_bchw, range_mode).to(meta["device"])
    y_m11 = attack_fn(x_m11, *args, **kwargs).clamp(-1.0, 1.0)

    y_bchw = _from_minus1_1(y_m11, range_mode, meta["dtype"]).to(meta["device"])
    y = _from_bchw(y_bchw, meta)
    return y


# ============================================================
# 2) Geometric Attacks
# ============================================================
def rotate_tensor(x: torch.Tensor, angle: float) -> torch.Tensor:
    def _core(z: torch.Tensor):
        return TF.rotate(
            z,
            angle=float(angle),
            interpolation=InterpolationMode.BILINEAR,
            expand=False,
            fill=-1.0,
        )

    return _apply_attack_preserve(x, _core)


def crop(x: torch.Tensor, pct: int, start_y: int = 50, start_x: int = 50) -> torch.Tensor:
    def _core(z: torch.Tensor):
        _, _, H, W = z.shape
        crop_side = int(min(H, W) * (pct / 100.0))
        crop_side = max(10, crop_side)

        if (start_y + crop_side > H) or (start_x + crop_side > W):
            return z

        cropped = z[:, :, start_y : start_y + crop_side, start_x : start_x + crop_side]
        return F.interpolate(cropped, size=(H, W), mode="bilinear", align_corners=False)

    return _apply_attack_preserve(x, _core)


def scaled(x: torch.Tensor, pct: int) -> torch.Tensor:
    def _core(z: torch.Tensor):
        _, _, H, W = z.shape
        scale = 1.0 + (pct / 100.0)
        bigger = F.interpolate(z, scale_factor=scale, mode="bilinear", align_corners=False)
        back = F.interpolate(bigger, size=(H, W), mode="bilinear", align_corners=False)
        return back

    return _apply_attack_preserve(x, _core)


def flipping(x: torch.Tensor, mode: str) -> torch.Tensor:
    def _core(z: torch.Tensor):
        m = str(mode).upper()
        if m == "H":
            return torch.flip(z, dims=[3])
        if m == "V":
            return torch.flip(z, dims=[2])
        if m == "B":
            return torch.flip(z, dims=[2, 3])
        return z

    return _apply_attack_preserve(x, _core)


def resized(x: torch.Tensor, pct: int) -> torch.Tensor:
    def _core(z: torch.Tensor):
        _, _, H, W = z.shape
        level_ratio = pct / 100.0
        down = max(0.2, 1.0 - level_ratio)  # 0.8/0.5/0.2
        new_h = max(1, int(H * down))
        new_w = max(1, int(W * down))
        small = F.interpolate(z, size=(new_h, new_w), mode="bilinear", align_corners=False)
        back = F.interpolate(small, size=(H, W), mode="bilinear", align_corners=False)
        return back

    return _apply_attack_preserve(x, _core)


# ============================================================
# 3) Signal Processing Attacks
# ============================================================
def jpeg_compression(x: torch.Tensor, quality: int) -> torch.Tensor:
    """
    JPEG compression عبر torchvision.io (بدون ملفات) مع دعم Batch.
    """
    def _core(z: torch.Tensor):
        device = z.device
        z_cpu = z.detach().cpu().clamp(-1.0, 1.0)

        B, C, H, W = z_cpu.shape
        outs = []

        q = int(quality)

        for i in range(B):
            # [-1,1] -> uint8 [0,255] بشكل CHW (torchvision expects CHW uint8)
            x01 = (z_cpu[i] + 1.0) / 2.0
            img_u8 = (x01 * 255.0).round().clamp(0, 255).to(torch.uint8)   # CHW uint8

            # encode -> bytes -> decode
            jpeg_bytes = tvio.encode_jpeg(img_u8, quality=q)
            dec_u8 = tvio.decode_jpeg(jpeg_bytes)  # CHW uint8

            # رجوع إلى [-1,1] float
            dec_f = dec_u8.to(torch.float32) / 255.0
            dec_m11 = (dec_f * 2.0 - 1.0).clamp(-1.0, 1.0)
            outs.append(dec_m11)

        out = torch.stack(outs, dim=0).to(device)  # BCHW [-1,1]
        return out

    return _apply_attack_preserve(x, _core)



def gaussian_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    sigma: نفس معنى كودك السابق (على مقياس 0..255).
    """
    def _core(z: torch.Tensor):
        z = z.clamp(-1.0, 1.0)
        sigma_norm = (float(sigma) / 255.0) * 2.0
        noise = torch.randn_like(z) * sigma_norm
        return (z + noise).clamp(-1.0, 1.0)

    return _apply_attack_preserve(x, _core)


def speckle_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    def _core(z: torch.Tensor):
        z = z.clamp(-1.0, 1.0)
        noise = torch.randn_like(z) * float(sigma)
        return (z + z * noise).clamp(-1.0, 1.0)

    return _apply_attack_preserve(x, _core)

def blurring(x: torch.Tensor, k: int) -> torch.Tensor:
    def _core(z: torch.Tensor):
        kk = int(k)
        if kk % 2 == 0:
            kk += 1

        if _HAS_KORNIA:
            # Kornia expects BCHW float
            sigma = float(kk) / 6.0
            # kernel_size must be (ky, kx), sigma can be (sy, sx)
            out = Kf.gaussian_blur2d(z, (kk, kk), (sigma, sigma))
            return out.clamp(-1.0, 1.0)

        # fallback torchvision
        sigma = float(kk) / 6.0
        out = TF.gaussian_blur(z, kernel_size=[kk, kk], sigma=[sigma, sigma])
        return out.clamp(-1.0, 1.0)

    return _apply_attack_preserve(x, _core)



def brightness(x: torch.Tensor, factor: float) -> torch.Tensor:
    def _core(z: torch.Tensor) -> torch.Tensor:
        z = z.clamp(-1.0, 1.0)

        # [-1,1] -> [0,1]
        z01 = (z + 1.0) / 2.0

        # apply brightness
        out01 = (z01 * float(factor)).clamp(0.0, 1.0)

        # [0,1] -> [-1,1]
        return (out01 * 2.0 - 1.0).clamp(-1.0, 1.0)

    return _apply_attack_preserve(x, _core)

def sharpness(x: torch.Tensor, amount: float = 1.0) -> torch.Tensor:
    def _core(z: torch.Tensor) -> torch.Tensor:
        z = z.clamp(-1.0, 1.0)

        # Simple unsharp mask: x + amount * (x - blur)
        blur = F.avg_pool2d(z, kernel_size=3, stride=1, padding=1)
        out = z + float(amount) * (z - blur)

        return out.clamp(-1.0, 1.0)

    return _apply_attack_preserve(x, _core)


def median_filtering(x: torch.Tensor, k: int) -> torch.Tensor:
    def _core(z: torch.Tensor):
        kk = int(k)
        if kk % 2 == 0:
            kk += 1

        z = z.clamp(-1.0, 1.0)

        if _HAS_KORNIA:
            out = Kf.median_blur(z, (kk, kk))
            return out.clamp(-1.0, 1.0)

        # fallback (unfold median)
        B, C, H, W = z.shape
        pad = kk // 2
        z_pad = F.pad(z, (pad, pad, pad, pad), mode="reflect")
        patches = z_pad.unfold(2, kk, 1).unfold(3, kk, 1)
        patches = patches.contiguous().view(B, C, H, W, kk * kk)
        median_vals = patches.median(dim=-1).values
        return median_vals.clamp(-1.0, 1.0)

    return _apply_attack_preserve(x, _core)


__all__ = [
    # geometric
    "rotate_tensor", "crop", "scaled", "flipping", "resized",
    # signal
    "jpeg_compression", "gaussian_noise", "speckle_noise",
    "blurring", "brightness", "sharpness", "median_filtering",
]
