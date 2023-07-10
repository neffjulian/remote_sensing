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

def transform_model_output(model_output: list) -> list[np.ndarray]:
    img_s2 = []
    img_sr = []
    img_ps = []
    names = []

    reconstructed_images = []

    normal_s2 = []
    normal_sr = []
    normal_ps = []
    normal_names = []
    for out in model_output:
        s2, y, ps, name = out

        if name[0].startswith("03_"):
            img_s2.append(s2)
            img_sr.append(y)
            img_ps.append(ps)
            names.append(name)
        else:
            normal_s2.append(s2)
            normal_sr.append(y)
            normal_ps.append(ps)
            normal_names.append(name)

    normal_result_s2 = torch.cat(normal_s2, dim=0).squeeze().numpy()
    normal_result_sr = torch.cat(normal_sr, dim=0).squeeze().numpy()
    normal_result_ps = torch.cat(normal_ps, dim=0).squeeze().numpy()
    normal_names = [string for tup in normal_names for string in tup]

    for i in range(0, len(normal_names)):
        reconstructed_images.append((normal_result_s2[i], normal_result_sr[i], normal_result_ps[i], normal_names[i]))
            
    result_s2 = torch.cat(img_s2, dim=0).squeeze().numpy()
    result_sr = torch.cat(img_sr, dim=0).squeeze().numpy()
    result_ps = torch.cat(img_ps, dim=0).squeeze().numpy()

    image_names = [string for tup in names for string in tup]
    enumerated_names = [(i, str) for i, str in enumerate(image_names)]
    sorted_names = sorted(enumerated_names, key=lambda x: x[1])
    print("Add reconstructed images")

    for i in range(0, len(sorted_names), 16):
        image_s2 = np.zeros((600, 600))
        image_sr = np.zeros((600, 600))
        image_ps = np.zeros((600, 600))
        for j in range(4):
            for k in range(4):
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
    print(output[0])
    print(type(output), len(output))
    for out in output:
        print(len(out), type(out))
    raise Exception
    transformed_output = transform_model_output(output)
    results = RESULT_DIR.joinpath(name)
    results.mkdir(parents=True, exist_ok=True)
    lr_hr_psnrs = []
    sr_hr_psnrs = []
    lr_hr_ssims = []
    sr_hr_ssims = []
    for i, out in enumerate(transformed_output):
        out_file = results.joinpath(out[3][:-4] + '.png')
        lr_hr_psnr = psnr(out[0], out[2])
        sr_hr_psnr = psnr(out[1], out[2])
        lr_hr_ssim, _ = ssim((out[0] * (255.0 / 8.0)).astype(np.uint8), (out[2] * (255.0 / 8.0)).astype(np.uint8), full=True)
        sr_hr_ssim, _ = ssim((out[1] * (255.0 / 8.0)).astype(np.uint8), (out[2] * (255.0 / 8.0)).astype(np.uint8), full=True)
        print(out_file.name, "LR-HR PSNR:", lr_hr_psnr, "  SR-HR PSNR:", sr_hr_psnr," |  LR-HR SSIM:", lr_hr_ssim, " SR-HR SSIM:", sr_hr_ssim)
        save_output_visualization(out[0], out[1], out[2], out_file)
        np.save(results.joinpath(out[3][:-4] + '.npy'), out[2])
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