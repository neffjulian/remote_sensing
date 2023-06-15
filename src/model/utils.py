import os
import gc
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
RESULT_DIR = Path(__file__).parent.parent.parent.joinpath('data', 'results')

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

    reconstructed_images = []

    image_names = [string for tup in names for string in tup]
    enumerated_names = [(i, str) for i, str in enumerate(image_names)]
    sorted_names = sorted(enumerated_names, key=lambda x: x[1])

    for i in range(0, len(sorted_names), 4):

        image_s2 = np.zeros((640, 640))
        image_s2[:320, :320] = result_s2[sorted_names[i][0]]
        image_s2[:320, 320:] = result_s2[sorted_names[i+1][0]]
        image_s2[320:, :320] = result_s2[sorted_names[i+2][0]]
        image_s2[320:, 320:] = result_s2[sorted_names[i+3][0]]

        image_sr = np.zeros((640, 640))
        image_sr[:320, :320] = result_sr[sorted_names[i][0]]
        image_sr[:320, 320:] = result_sr[sorted_names[i+1][0]]
        image_sr[320:, :320] = result_sr[sorted_names[i+2][0]]
        image_sr[320:, 320:] = result_sr[sorted_names[i+3][0]]

        image_ps = np.zeros((640, 640))
        image_ps[:320, :320] = result_ps[sorted_names[i][0]]
        image_ps[:320, 320:] = result_ps[sorted_names[i+1][0]]
        image_ps[320:, :320] = result_ps[sorted_names[i+2][0]]
        image_ps[320:, 320:] = result_ps[sorted_names[i+3][0]]

        reconstructed_images.append((image_s2, image_sr, image_ps))
    
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
        out_file = results.joinpath(f"{i:04d}")
        print(out_file)
        save_output_visualization(out[0], out[1], out[2], out_file)

def report_gpu():
   print(torch.cuda.list_gpu_processes())
   gc.collect()
   torch.cuda.empty_cache()

if __name__ == "__main__":
    report_gpu()