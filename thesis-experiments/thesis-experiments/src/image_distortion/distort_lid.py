import json
import os
from typing import Any, Callable

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as NN_F
import torchvision.transforms.functional as F

DistortionFn = Callable[[PIL.Image.Image, dict[str, Any]], PIL.Image.Image]
DISTORTION_REGISTRY: dict[str, DistortionFn] = {}


def register_distortion(name: str) -> Callable[[DistortionFn], DistortionFn]:
    def decorator(fn: DistortionFn) -> DistortionFn:
        DISTORTION_REGISTRY[name] = fn
        return fn
    return decorator


@register_distortion("vcd_diffusion_noise")
def vcd_diffusion_noise(
    image: PIL.Image.Image, parameters: dict[str, Any]
) -> PIL.Image.Image:
    """Forward diffusion noise (exact VCD repo implementation)."""
    T = int(parameters.get("T", 500))
    if T == 0:
        return image

    num_steps = 1000
    betas = torch.linspace(-6, 6, num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    def q_x(x_0, t):
        noise = torch.randn_like(x_0)
        return alphas_bar_sqrt[t] * x_0 + one_minus_alphas_bar_sqrt[t] * noise

    img_tensor = F.to_tensor(image)
    noise_step = min(T, num_steps - 1)
    noisy = img_tensor.clone()
    image_tensor_cd = q_x(noisy, noise_step)
    image_tensor_cd = torch.clamp(image_tensor_cd, 0.0, 1.0)
    return F.to_pil_image(image_tensor_cd)


@register_distortion("speckle_noise")
def speckle_noise(
    image: PIL.Image.Image, parameters: dict[str, float]
) -> PIL.Image.Image:
    """Multiplicative speckle noise"""
    speckle_intensity = parameters.get("speckle_intensity", 0.5)
    img_tensor = F.to_tensor(image)
    noise = torch.randn_like(img_tensor) * speckle_intensity
    speckled = img_tensor + (img_tensor * noise)
    speckled = torch.clamp(speckled, 0.0, 1.0)
    return F.to_pil_image(speckled)


@register_distortion("color_mae_masking")
def color_mae_masking(
    image: PIL.Image.Image, parameters: dict[str, Any]
) -> PIL.Image.Image:
    """ColorMAE masking"""
    mask_ratio = parameters.get("mask_ratio", 0.75)
    patch_size = parameters.get("patch_size", 64)
    color = parameters.get("color", "white")

    W_img, H_img = image.size
    H_p, W_p = H_img // patch_size, W_img // patch_size
    if H_p == 0 or W_p == 0:
        H_p, W_p = 1, 1

    # FFT-based color noise
    noise = torch.randn(H_p, W_p)
    F_noise = torch.fft.fftshift(torch.fft.fft2(noise))
    y = torch.linspace(-1, 1, H_p).view(-1, 1).repeat(1, W_p)
    x = torch.linspace(-1, 1, W_p).view(1, -1).repeat(H_p, 1)
    f = torch.sqrt(x**2 + y**2) + 1e-8

    if color == "white":
        S = torch.ones_like(f)
    elif color == "red":
        S = 1 / (f**1.5)
    elif color == "blue":
        S = f**1.5
    elif color == "green":
        S = torch.exp(-((f - 0.4) ** 2) / 0.05)
    elif color == "purple":
        S = f**2.5
    else:
        S = torch.ones_like(f)

    S[H_p // 2, W_p // 2] = 0
    F_colored = F_noise * S
    noise_grid = torch.fft.ifft2(torch.fft.ifftshift(F_colored)).real

    L = H_p * W_p
    len_keep = int(L * (1 - mask_ratio))
    window_flat = noise_grid.reshape(1, -1)
    ids_shuffle = torch.argsort(window_flat, dim=1, descending=True)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    mask = torch.ones([1, L])
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    patch_mask = mask.view(1, 1, H_p, W_p)
    pixel_mask = NN_F.interpolate(
        patch_mask, size=(H_img, W_img), mode="nearest"
    ).squeeze()
    visual_mask = 1.0 - pixel_mask
    img_tensor = F.to_tensor(image)
    masked = img_tensor * visual_mask.unsqueeze(0)
    return F.to_pil_image(masked)


LID_IMAGES_DIR = "/mnt/storagecube/filipe/lid/images"
LID_DISTORTED_BASE = "/mnt/storagecube/filipe/lid"

# Distortion jobs
DISTORTION_JOBS = [
    (
        "speckle_noise",
        "speckle_noise",
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        lambda lvl: {"speckle_intensity": lvl},
    ),
    (
        "vcd_diffusion_noise",
        "vcd_diffusion_noise",
        [0, 203, 406, 592, 795, 999],
        lambda lvl: {"T": int(lvl)},
    ),
    (
        "color_mae_masking",
        "color_mae_masking/blue",
        [0.0, 0.25, 0.5, 0.75, 1.0],
        lambda lvl: {"color": "blue", "mask_ratio": lvl},
    ),
    (
        "color_mae_masking",
        "color_mae_masking/green",
        [0.0, 0.25, 0.5, 0.75, 1.0],
        lambda lvl: {"color": "green", "mask_ratio": lvl},
    ),
    (
        "color_mae_masking",
        "color_mae_masking/purple",
        [0.0, 0.25, 0.5, 0.75, 1.0],
        lambda lvl: {"color": "purple", "mask_ratio": lvl},
    ),
    (
        "color_mae_masking",
        "color_mae_masking/red",
        [0.0, 0.25, 0.5, 0.75, 1.0],
        lambda lvl: {"color": "red", "mask_ratio": lvl},
    ),
]


if __name__ == "__main__":
    LID_JSON = os.path.join(LID_IMAGES_DIR, "..", "new_llava_AeroCaps_short_objects.json")
    hal_images: set[str] = set()
    with open(LID_JSON, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            if sample.get("obj_hal") == 1:
                hal_images.add(sample["image_name"])

    all_images = sorted(
        f for f in os.listdir(LID_IMAGES_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    image_filenames = [f for f in all_images if f in hal_images]
    skipped = len(all_images) - len(image_filenames)
    print(f"Total images: {len(all_images)} | with hallucinations: {len(image_filenames)} (skipped {skipped})")

    total_jobs = sum(len(levels) for _, _, levels, _ in DISTORTION_JOBS)
    job_idx = 0

    for distortion_name, output_subdir, levels, params_fn in DISTORTION_JOBS:
        distort_fn = DISTORTION_REGISTRY[distortion_name]
        base_out = os.path.join(LID_DISTORTED_BASE, output_subdir)

        for level in levels:
            job_idx += 1
            level_dir = os.path.join(base_out, f"level_{level}")
            os.makedirs(level_dir, exist_ok=True)

            params = params_fn(level)
            print(f"\n[{job_idx}/{total_jobs}] {distortion_name} | "
                  f"output={output_subdir}/level_{level} | params={params}")

            already_done = os.listdir(level_dir)
            if len(already_done) == len(image_filenames):
                print(f"  -> all {len(image_filenames)} images already exist, skipping")
                continue

            for i, filename in enumerate(image_filenames, 1):
                out_path = os.path.join(level_dir, filename)
                if os.path.exists(out_path):
                    continue  # resume partial runs

                img = PIL.Image.open(os.path.join(LID_IMAGES_DIR, filename))
                distorted = distort_fn(img, params)
                distorted.save(out_path)

                if i % 100 == 0 or i == len(image_filenames):
                    print(f"  [{i}/{len(image_filenames)}]")

            print(f"  -> done ({distortion_name}, level={level})")

    print(f"\nAll {total_jobs} distortion × level jobs complete.")
