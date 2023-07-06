import os
import gc
from pathlib import Path
from math import sqrt, log10

import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
RESULT_DIR = Path(__file__).parent.parent.parent.joinpath('data', 'results')

def psnr(x, y):
    return 20 * log10(8. / sqrt(np.mean((x - y) ** 2)))

def transform_model_output(model_output: list) -> list[np.ndarray]:
    img_s2 = []
    img_sr = []
    img_ps = []
    names = []
    for out in model_output:
        s2, y, ps, name = out

        img_s2.append(s2)
        img_sr.append(y)
        img_ps.append(ps)
        names.append(name)

    result_s2 = torch.cat(img_s2, dim=0).squeeze().numpy()
    result_sr = torch.cat(img_sr, dim=0).squeeze().numpy()
    result_ps = torch.cat(img_ps, dim=0).squeeze().numpy()

    image_names = [string for tup in names for string in tup]
    enumerated_names = [(i, str) for i, str in enumerate(image_names)]
    sorted_names = sorted(enumerated_names, key=lambda x: x[1])

    tiles = int(len(sorted_names) / 48)
    sqrt_tiles = int(sqrt(tiles))

    reconstructed_images = []
    for i in range(0, len(sorted_names), tiles):
        image_s2 = np.zeros((600, 600))
        image_sr = np.zeros((600, 600))
        image_ps = np.zeros((600, 600))
        for j in range(sqrt_tiles):
            for k in range(sqrt_tiles):
                image_s2[(j*150):((j+1)*150), (k*150):((k+1)*150)] = result_s2[sorted_names[i + j*4 + k][0]]
                image_sr[(j*150):((j+1)*150), (k*150):((k+1)*150)] = result_sr[sorted_names[i + j*4 + k][0]]
                image_ps[(j*150):((j+1)*150), (k*150):((k+1)*150)] = result_ps[sorted_names[i + j*4 + k][0]]

        reconstructed_images.append((image_s2, image_sr, image_ps, sorted_names[i][1]))
    return reconstructed_images

def save_output_visualization(sentinel_2: np.ndarray, super_resolved: np.ndarray, planet_scope: np.ndarray, dir: Path):
    f, axes = plt.subplots(1, 3, figsize=(30, 10))

    # Plot for S2 image
    ax1 = axes[0]
    im1 = ax1.imshow(sentinel_2, cmap='viridis', vmin=0., vmax=8.)
    ax1.set_title('Sentinel-2 Image')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1)

    # Plot for SR image
    ax2 = axes[1]
    im2 = ax2.imshow(super_resolved, cmap='viridis', vmin=0., vmax=8.)
    ax2.set_title('Super-Resolved Image')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_aspect('equal')
    plt.colorbar(im2, ax=ax2)

    # Plot for PlanetScope image
    ax3 = axes[2]
    im3 = ax3.imshow(planet_scope, cmap='viridis', vmin=0., vmax=8.)
    ax3.set_title('PlanetScope Image')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_aspect('equal')
    plt.colorbar(im3, ax=ax3, label=r'LAI [m$^2$ m$^{-2}$]')

    plt.tight_layout()
    plt.savefig(dir, dpi=300)
    plt.close(f)

def visualize_output(name: str, output: list) -> None:
    transformed_output = transform_model_output(output)
    results = RESULT_DIR.joinpath(name)
    results.mkdir(parents=True, exist_ok=True)
    for i, out in enumerate(transformed_output):
        out_file = results.joinpath(out[3][:-4] + '.png')
        lr_hr_psnr = psnr(out[0], out[2])
        sr_hr_psnr = psnr(out[1], out[2])
        lr_hr_ssim, _ = ssim((out[0] * (255.0 / 8.0)).astype(np.uint8), (out[2] * (255.0 / 8.0)).astype(np.uint8), full=True)
        sr_hr_ssim, _ = ssim((out[1] * (255.0 / 8.0)).astype(np.uint8), (out[2] * (255.0 / 8.0)).astype(np.uint8), full=True)
        print(out_file.name, "LR-HR PSNR:", lr_hr_psnr, "  SR-HR PSNR:", sr_hr_psnr," |  LR-HR SSIM:", lr_hr_ssim, " SR-HR SSIM:", sr_hr_ssim)
        save_output_visualization(out[0], out[1], out[2], out_file)
        np.save(out[2], results.joinpath(out[3][:-4] + '.npy'))

def report_gpu():
   print(torch.cuda.list_gpu_processes())
   gc.collect()
   torch.cuda.empty_cache()

if __name__ == "__main__":
    report_gpu()