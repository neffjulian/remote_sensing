from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

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