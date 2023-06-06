import os

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
RESULT_DIR = Path().absolute().parent.parent.joinpath('data', 'results')

def transform_model_output(model_output: list) -> list[np.ndarray]:
    img_original = []
    img_sr = []
    for out in model_output:
        x, y = out
        img_sr.append(x)
        img_original.append(y)

    result_sr = torch.cat(img_sr, dim=0).squeeze().numpy()
    result_or = torch.cat(img_original, dim=0).squeeze().numpy()

    assert result_sr.shape[0] % 4 == 0 and result_or.shape[0] % 4 == 0

    reconstructed_images = []
    num_images  = int(result_sr.shape[0] / 4)
    for i in range(num_images):
        image_sr = np.zeros((640, 640))
        image_sr[:320, :320] = result_sr[(i*4)]
        image_sr[:320, 320:] = result_sr[(i*4) + 1]
        image_sr[320:, :320] = result_sr[(i*4) + 2]
        image_sr[320:, 320:] = result_sr[(i*4) + 3]

        image_or = np.zeros((640, 640))
        image_or[:320, :320] = result_or[(i*4)]
        image_or[:320, 320:] = result_or[(i*4) + 1]
        image_or[320:, :320] = result_or[(i*4) + 2]
        image_or[320:, 320:] = result_or[(i*4) + 3]

        reconstructed_images.append((image_sr, image_or))
    
    return reconstructed_images

def save_output_visualization(super_resolved: np.ndarray, original: np.ndarray, dir: Path):
    f, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Plot for SR image
    ax1 = axes[0]
    im1 = ax1.imshow(super_resolved, cmap='viridis', vmin=0., vmax=8.)
    ax1.set_title('Super-Resolved Image')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1)

    # Plot for original image
    ax2 = axes[1]
    im2 = ax2.imshow(original, cmap='viridis', vmin=0., vmax=8.)
    ax2.set_title('Original Image')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_aspect('equal')
    plt.colorbar(im2, ax=ax2, label=r'LAI [m$^2$ m$^{-2}$]')

    plt.tight_layout()
    plt.savefig(dir, dpi=300)
    plt.close(f)

def visualize_output(name: str, output: list) -> None:
    transformed_output = transform_model_output(output)
    current_date = datetime.now()
    date_string = current_date.strftime("%Y_%m_%d_%H_%M_%S")
    results = RESULT_DIR.joinpath(f"{name}_{date_string}")
    results.mkdir()
    print(len(transformed_output))
    for i, out in enumerate(transformed_output):
        out_file = results.joinpath(f"{i:04d}")
        print(out_file)
        save_output_visualization(out[0], out[1], out_file)

def save_tile_visualization(tile: np.ndarray, dir: Path):
    f, ax = plt.subplots(1, 1, figsize=(10, 10))
    tile.plot(
        colormap='viridis',
        colorbar_label=r'LAI [m$^2$ m$^{-2}$]',
        vmin=0.,
        vmax=8.,
        ax=ax,
        fontsize=18
    )
    ax.set_title('')
    f.savefig(dir, dpi=300)
    plt.close(f)