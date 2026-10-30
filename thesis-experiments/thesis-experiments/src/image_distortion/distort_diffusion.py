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
    """
    Applies visual uncertainty using the exact forward diffusion implementation
    from the official VCD GitHub repository.
    """
    T = int(parameters.get("T", 500))
    if T == 0:
        return image

    num_steps = 1000

    betas = torch.linspace(-6, 6, num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_prod_p = torch.cat(
        [torch.tensor([1]).float(), alphas_prod[:-1]], 0
    )  
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    def q_x(x_0, t):
        noise = torch.randn_like(x_0)
        alphas_t = alphas_bar_sqrt[t]
        alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
        return alphas_t * x_0 + alphas_1_m_t * noise

    img_tensor = F.to_tensor(image)

    noise_step = min(T, num_steps - 1)

    noisy_image = img_tensor.clone()
    image_tensor_cd = q_x(noisy_image, noise_step)

    image_tensor_cd = torch.clamp(image_tensor_cd, 0.0, 1.0)

    return F.to_pil_image(image_tensor_cd)


@register_distortion("speckle_noise")
def speckle_noise(
    image: PIL.Image.Image, parameters: dict[str, float]
) -> PIL.Image.Image:
    """Applies multiplicative speckle noise to simulate visual uncertainty."""
    speckle_intensity = parameters.get("speckle_intensity", 0.5)
    img_tensor = F.to_tensor(image)

    # Speckle noise: I' = I + I * n, where n ~ N(0, speckle_intensity^2)
    noise = torch.randn_like(img_tensor) * speckle_intensity
    speckled_img_tensor = img_tensor + (img_tensor * noise)

    speckled_img_tensor = torch.clamp(speckled_img_tensor, 0.0, 1.0)
    speckled_img = F.to_pil_image(speckled_img_tensor)

    return speckled_img


@register_distortion("color_mae_masking")
def color_mae_masking(
    image: PIL.Image.Image, parameters: dict[str, Any]
) -> PIL.Image.Image:
    """
    Applies ColorMAE masking
    """
    mask_ratio = parameters.get("mask_ratio", 0.75)
    patch_size = parameters.get("patch_size", 64)
    color = parameters.get("color", "white")

    W_img, H_img = image.size
    H_p, W_p = H_img // patch_size, W_img // patch_size

    if H_p == 0 or W_p == 0:
        H_p, W_p = 1, 1

    noise = torch.randn(H_p, W_p)
    F_noise = torch.fft.fftshift(torch.fft.fft2(noise))

    y = torch.linspace(-1, 1, H_p).view(-1, 1).repeat(1, W_p)
    x = torch.linspace(-1, 1, W_p).view(1, -1).repeat(H_p, 1)
    f = torch.sqrt(x**2 + y**2) + 1e-8

    if color == "white":
        S = torch.ones_like(f)
    elif color == "red":
        S = 1 / (f**1.5)  # Low-pass filter
    elif color == "blue":
        S = f**1.5  # High-pass filter highly scattered dots
    elif color == "green":
        S = torch.exp(-((f - 0.4) ** 2) / 0.05)  # Band-pass
    elif color == "purple":
        S = f**2.5  # Extreme high-pass 
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

    # Apply the mask 
    patch_mask = mask.view(1, 1, H_p, W_p)
    pixel_mask = NN_F.interpolate(
        patch_mask, size=(H_img, W_img), mode="nearest"
    ).squeeze()

    visual_mask = 1.0 - pixel_mask
    img_tensor = F.to_tensor(image)
    masked_img_tensor = img_tensor * visual_mask.unsqueeze(0)

    return F.to_pil_image(masked_img_tensor)


if __name__ == "__main__":
    original_images_path = (
        "/mnt/storagecube/filipe/rspope/rspope_evaluation/rspope_image/original/complete"
    )

    distortion = "vcd_diffusion_noise"  # Change this to apply different distortions

    # Scale T from 0 to 999 using integer steps 
    distortion_levels = [999]
    base_output_path = f"/mnt/storagecube/filipe/rspope/rspope_evaluation/rspope_image/vcd_diffusion_noise/complete"



    for level in distortion_levels:
        print(f"Processing {distortion} level (T): {level}")
        output_path = os.path.join(base_output_path, f"level_{level}")
        os.makedirs(output_path, exist_ok=True)

        for i, filename in enumerate(os.listdir(original_images_path)):
            print(
                f"Distorting image: {i + 1}/{len(os.listdir(original_images_path))} - {filename}"
            )
            if filename.endswith(".png") or filename.endswith(".jpg"):
                img = PIL.Image.open(os.path.join(original_images_path, filename))

                distort_fn = DISTORTION_REGISTRY[distortion]
                distorted_img = distort_fn(img, {"T": level})

                distorted_img.save(os.path.join(output_path, filename))

        print(f"Finished processing {distortion} level: {level}\n")
    print("Done!\n")
