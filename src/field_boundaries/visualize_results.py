import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

VALIDATE_DIR = Path(__file__).parent.parent.parent.joinpath("data", "validate")

# def plot(data: tuple[float, float], out: Path):
#     baseline_values = [x[0] for x in data]
#     sr_values = [x[1] for x in data]

#     indices = range(len(data))

#     plt.plot(indices, baseline_values, 'bo', markersize=2, label='Bicubic')
#     plt.plot(indices, sr_values, 'ro', markersize=2, label='Super Resolution')

#     plt.xlabel('Index')
#     plt.ylabel('Value')
#     plt.legend()
#     plt.show()

def create_results() -> None:
    sentinel_bands = ["10m", "20m"]
    planetscope_bands = ["4b", "8b"]
    models = ["srcnn", "edsr", "rrdb"]
    baseline = "bicubic"

    header = ["s2_band", "ps_band", "month", "model", "nr_corr_coeff", "corr_coeff_baseline", 
              "corr_coeff", "nr_psnr", "psnr_baseline", "psnr", "nr_ssim", "ssim_baseline", "ssim"]
    results = []

    for s2_band in sentinel_bands:
        for ps_band in planetscope_bands:
            for model in models:
                sr_dir = VALIDATE_DIR.joinpath(f"{s2_band}_{model}")

                for month in sr_dir.iterdir():
                    baseline_csv = VALIDATE_DIR.joinpath(f"{s2_band}_{baseline}", month.name, f"{ps_band}_results.csv")
                    sr_csv = month.joinpath(f"{ps_band}_results.csv")

                    baseline_df = pd.read_csv(baseline_csv)
                    sr_df = pd.read_csv(sr_csv)

                    corr_coeff, psnr, ssim = get_results(baseline_df, sr_df)

                    results.append([s2_band, ps_band, month.name, model, corr_coeff[0], corr_coeff[1], corr_coeff[2], 
                                    psnr[0], psnr[1], psnr[2], ssim[0], ssim[1], ssim[2]])
    
    results_df = pd.DataFrame(results, columns=header)
    results_df.to_csv(VALIDATE_DIR.joinpath("results.csv"), index=False)

def get_results(baseline: pd.DataFrame, sr: pd.DataFrame) -> tuple:
    assert len(baseline) == len(sr), "Dataframes do not have the same length."
    baseline_corr_coeff, sr_corr_coeff = [], []
    baseline_psnr, sr_psnr = [], []
    baseline_ssim, sr_ssim = [], []
    
    for (row1, row2) in zip(baseline.itertuples(index=False), sr.itertuples(index=False)):
        # row = {index, boundary, corr_coeff, p_value, psnr, ssim}

        # Add Correlation Coefficient only if p_value below 5%
        if not (math.isnan(row1[2]) or math.isnan(row2[2])) and (row1[3] < 0.05 or row2[3] < 0.05):
            baseline_corr_coeff.append(row1[2])
            sr_corr_coeff.append(row2[2])
        if not (math.isnan(row1[4]) or math.isnan(row2[4])): # PSNR
            baseline_psnr.append(row1[4])
            sr_psnr.append(row2[4])
        if not (math.isnan(row1[5]) or math.isnan(row2[5])): # SSIM
            baseline_ssim.append(row1[5])
            sr_ssim.append(row2[5])

    corr_coeff = (len(baseline_corr_coeff), np.mean(baseline_corr_coeff), np.mean(sr_corr_coeff))
    psnr = (len(baseline_psnr), np.mean(baseline_psnr), np.mean(sr_psnr))
    ssim = (len(baseline_ssim), np.mean(baseline_ssim), np.mean(sr_ssim))
    return corr_coeff, psnr, ssim

def visualize(model: str, s2_band: str, ps_band: str):
    df = pd.read_csv(VALIDATE_DIR.joinpath("results.csv"))
    filtered_df = df.loc[(df["s2_band"] == s2_band) & (df["ps_band"] == ps_band)]
    months = ['03_mar', '04_apr', '05_may', '06_jun', '07_jul', '08_aug', '09_sep']

    corr_coeff = []
    psnr = []
    ssim = []

    baseline_corr_coeff = []
    baseline_psnr = []
    baseline_ssim = []

    for month in months:
        corr_coeff.append(filtered_df.loc[(filtered_df["month"] == month) & (filtered_df["model"] == model), "corr_coeff"].values[0])
        baseline_corr_coeff.append(filtered_df.loc[(filtered_df["month"] == month) & (filtered_df["model"] == model), "corr_coeff_baseline"].values[0])

        psnr.append(filtered_df.loc[(filtered_df["month"] == month) & (filtered_df["model"] == model), "psnr"].values[0])
        baseline_psnr.append(filtered_df.loc[(filtered_df["month"] == month) & (filtered_df["model"] == model), "psnr_baseline"].values[0])

        ssim.append(filtered_df.loc[(filtered_df["month"] == month) & (filtered_df["model"] == model), "ssim"].values[0])
        baseline_ssim.append(filtered_df.loc[(filtered_df["month"] == month) & (filtered_df["model"] == model), "ssim_baseline"].values[0])


    out_dir = VALIDATE_DIR.joinpath("visualize")
    out_dir.mkdir(parents=True, exist_ok=True)
    months = ["March", "April", "May", "June", "July", "August", "September"]


    # Correlation Coefficient
    out_name = f"correlation_coefficient_{s2_band}_{ps_band}_{model}"
    out_path = out_dir.joinpath(out_name)

    plt.plot(baseline_corr_coeff, label='Bicubic', marker='o', linestyle='-', color='blue')
    plt.plot(corr_coeff, label=model.upper(), marker='o', linestyle='-', color='red')

    plt.xlabel('Month')
    plt.ylabel('Correlation Coefficient')
    plt.ylim(0.7, 1.0)
    plt.title(f'Correlation Coefficient for {s2_band} and {ps_band} - {model.upper()}')
    plt.xticks(np.arange(len(months)), months)
    plt.legend()

    plt.savefig(out_path.with_suffix(".png"))
    plt.close()

    # PSNR
    out_name = f"psnr_{s2_band}_{ps_band}_{model}"
    out_path = out_dir.joinpath(out_name)

    plt.plot(baseline_psnr, label='Bicubic', marker='o', linestyle='-', color='blue')
    plt.plot(psnr, label=model.upper(), marker='o', linestyle='-', color='red')

    plt.xlabel('Month')
    plt.ylabel('PSNR')
    plt.ylim(20, 35)
    plt.title(f'PSNR for {s2_band} and {ps_band} - {model.upper()}')
    plt.xticks(np.arange(len(months)), months)
    plt.legend()

    plt.savefig(out_path.with_suffix(".png"))
    plt.close()

    # SSIM
    out_name = f"ssim_{s2_band}_{ps_band}_{model}"
    out_path = out_dir.joinpath(out_name)

    plt.plot(baseline_ssim, label='Bicubic', marker='o', linestyle='-', color='blue')
    plt.plot(ssim, label=model.upper(), marker='o', linestyle='-', color='red')

    plt.xlabel('Month')
    plt.ylabel('SSIM')
    plt.ylim(0.7, 1.0)
    plt.title(f'SSIM for {s2_band} and {ps_band} - {model.upper()}')
    plt.xticks(np.arange(len(months)), months)
    plt.legend()

    plt.savefig(out_path.with_suffix(".png"))
    plt.close()

def main():
    # create_results()
    models = ["srcnn", "edsr", "rrdb"]
    s2_bands = ["10m", "20m"]
    ps_bands = ["4b", "8b"]

    for model in models:
        for s2_band in s2_bands:
            for ps_band in ps_bands:
                visualize(model, s2_band, ps_band)

    # # 10 m vs 20 m
    # for model in models:
    #     corr_coeff_10m = results_df.loc[(results_df["s2_band"] == "10m") & (results_df["model"] == model), "corr_coeff"].values
    #     corr_coeff_20m = results_df.loc[(results_df["s2_band"] == "20m") & (results_df["model"] == model), "corr_coeff"].values

    #     # TODO
    
    # # 4b vs 8b
    # for model in models:
    #     corr_coeff_4b = results_df.loc[(results_df["ps_band"] == "4b") & (results_df["model"] == model), "corr_coeff"].values
    #     corr_coeff_8b = results_df.loc[(results_df["ps_band"] == "8b") & (results_df["model"] == model), "corr_coeff"].values
    
                    
if __name__ == '__main__':
    main()