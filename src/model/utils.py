import os
import gc
from pathlib import Path
from math import sqrt, log10
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
RESULT_DIR = Path(__file__).parent.parent.parent.joinpath('data', 'results')

def psnr(x, y):
    return 20 * log10(8. / sqrt(np.mean((x - y) ** 2)))

def transform_model_output(model_output: list, s2: bool) -> list[np.ndarray]:
    reconstructed_images = []
    for i in range(0, 2):
        name = model_output[i + 16][3][:-2]
        if s2:
            name += "s2"
        else:
            name += "ps"

        image_s2 = np.zeros((520, 520))
        image_sr = np.zeros((520, 520))
        image_ps = np.zeros((520, 520))
        for j in range(4):
            for k in range(4):
                if j == 0:
                    x_start = 0
                    x_end = 150
                elif j == 1:
                    x_start = 110
                    x_end = 260
                elif j == 2:
                    x_start = 260
                    x_end = 310
                elif j == 3:
                    x_start = 370
                    x_end = 520

                if k == 0:
                    y_start = 0
                    y_end = 150
                elif k == 1:
                    y_start = 110
                    y_end = 260
                elif k == 2:
                    y_start = 260
                    y_end = 310
                elif k == 3:
                    y_start = 370
                    y_end = 520

                image_s2[x_start:x_end, y_start:y_end] = model_output[i + j*4 + k][0]
                image_sr[x_start:x_end, y_start:y_end] = model_output[i + j*4 + k][1]
                image_ps[x_start:x_end, y_start:y_end] = model_output[i + j*4 + k][2]
        
        reconstructed_images.append((image_s2, image_sr, image_ps, name))
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
    lr, sr, hr, names = [], [], [], []
    for out in output:
        lr += out[0]
        sr += out[1]
        hr += out[2]
        names += out[3]
    outputs = []
    for i in range(len(names)):
        outputs.append(
            (torch.squeeze(lr[i]).numpy(), 
             torch.squeeze(sr[i]).numpy(), 
             torch.squeeze(hr[i]).numpy(), 
             names[i][:-4]))
    in_situ = sorted(outputs[:-64], key=lambda x: x[3])
    ps_sr = sorted(outputs[-64:-32], key=lambda x: x[3])
    s2_sr = sorted(outputs[-32:], key=lambda x: x[3])


    transformer_ps_sr = transform_model_output(ps_sr, False)
    transformer_s2_sr = transform_model_output(s2_sr, True)

    print(transformer_ps_sr[0][3], transformer_ps_sr[1][3])
    print(transformer_s2_sr[0][3], transformer_s2_sr[1][3])

    # transformed_output = in_situ + transformer_ps_sr + transformer_s2_sr
    transformed_output = transformer_ps_sr + transformer_s2_sr

    results = RESULT_DIR.joinpath(name)
    results.mkdir(parents=True, exist_ok=True)
    lr_hr_psnrs = []
    sr_hr_psnrs = []
    lr_hr_ssims = []
    sr_hr_ssims = []
    for i, out in enumerate(transformed_output):
        out_file = results.joinpath(out[3] + '.png')
        print(out_file, out[0].shape, out[1].shape, out[2].shape)
        lr_hr_psnr = psnr(out[0], out[2])
        sr_hr_psnr = psnr(out[1], out[2])
        lr_hr_ssim, _ = ssim((out[0] * (255.0 / 8.0)).astype(np.uint8), (out[2] * (255.0 / 8.0)).astype(np.uint8), full=True)
        sr_hr_ssim, _ = ssim((out[1] * (255.0 / 8.0)).astype(np.uint8), (out[2] * (255.0 / 8.0)).astype(np.uint8), full=True)
        print(out_file.name, "LR-HR PSNR:", lr_hr_psnr, "  SR-HR PSNR:", sr_hr_psnr," |  LR-HR SSIM:", lr_hr_ssim, " SR-HR SSIM:", sr_hr_ssim)
        save_output_visualization(out[0], out[1], out[2], out_file)
        np.save(results.joinpath(out[3] + '.npy'), out[2])
        lr_hr_psnrs.append(lr_hr_psnr)
        sr_hr_psnrs.append(sr_hr_psnr)
        lr_hr_ssims.append(lr_hr_ssim)
        sr_hr_ssims.append(sr_hr_ssim)

    print("--------------------- MEAN ------------------------")
    print("LR-HR PSNR:", np.mean(lr_hr_psnrs), "  SR-HR PSNR:", np.mean(sr_hr_psnrs), " (", \
          round(np.mean(sr_hr_psnrs) / np.mean(lr_hr_psnrs), 2),") | LR-HR SSIM:", np.mean(lr_hr_ssims), \
            " SR-HR SSIM:", np.mean(sr_hr_ssims), " (",round(np.mean(sr_hr_ssims) / np.mean(lr_hr_ssims) * 100,2),")")
    print("-------------------- MEDIAN -----------------------")
    print("LR-HR PSNR:", np.median(lr_hr_psnrs), "  SR-HR PSNR:", np.median(sr_hr_psnrs), " (", \
          round(np.mean(sr_hr_psnrs) / np.mean(lr_hr_psnrs), 2),") |  LR-HR SSIM:", np.median(lr_hr_ssims), \
            " SR-HR SSIM:", np.median(sr_hr_ssims), " (",round(np.median(sr_hr_ssims) / np.median(lr_hr_ssims) * 100,2),")")

def report_gpu():
   print(torch.cuda.list_gpu_processes())
   gc.collect()
   torch.cuda.empty_cache()

if __name__ == "__main__":
    report_gpu()