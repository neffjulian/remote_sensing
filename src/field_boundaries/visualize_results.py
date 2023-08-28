"""
Creates and visualizes plots using the output from validate_boundaries.py.

@date: 2023-08-28
@author: Julian Neff, ETH Zurich

Copyright (C) 2023 Julian Neff

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

VALIDATE_DIR = Path(__file__).parent.parent.parent.joinpath("data", "validate")

def create_results() -> None:
    """
    Create a summary CSV file containing calculated metrics across multiple configurations.
    
    This function iterates through all the combinations of Sentinel-2 and PlanetScope band 
    specifications, as well as the super-resolution models used. It reads the corresponding 
    validation results for each configuration and calculates summary metrics. The summary 
    is then saved into a CSV file.
    
    Output:
        Creates a summary CSV file in the VALIDATE_DIR with metrics for each configuration.
    """

    # Define the possible Sentinel-2 and PlanetScope bands, and super-resolution models.
    sentinel_bands = ["10m", "20m"]
    planetscope_bands = ["4b", "8b"]
    models = ["srcnn", "edsr", "rrdb", "esrgan", "bicubic"]
    baseline = "bicubic"

    # Header for the summary CSV file
    header = ["s2_band", "ps_band", "month", "model", "nr_corr_coeff", "corr_coeff_baseline", 
              "corr_coeff", "nr_psnr", "psnr_baseline", "psnr", "nr_ssim", "ssim_baseline", "ssim"]
    
    # Initialize the list to store the results
    results = []

    # Loop through each combination of configurations
    for s2_band in sentinel_bands:
        for ps_band in planetscope_bands:
            for model in models:
                sr_dir = VALIDATE_DIR.joinpath(f"{s2_band}_{model}")

                # Iterate through each month for which validation was done
                for month in sr_dir.iterdir():
                    # Locate the baseline and model-specific result CSVs
                    baseline_csv = VALIDATE_DIR.joinpath(f"{s2_band}_{baseline}", month.name, f"{ps_band}_results.csv")
                    sr_csv = month.joinpath(f"{ps_band}_results.csv")

                    # Read the baseline and model-specific result data
                    baseline_df = pd.read_csv(baseline_csv)
                    sr_df = pd.read_csv(sr_csv)

                    # Calculate summary metrics
                    corr_coeff, psnr, ssim = get_results(baseline_df, sr_df)

                    # Store the results
                    results.append([s2_band, ps_band, month.name, model, corr_coeff[0], corr_coeff[1], corr_coeff[2], 
                                    psnr[0], psnr[1], psnr[2], ssim[0], ssim[1], ssim[2]])
    
    # Create a DataFrame from the results and save it as a CSV
    results_df = pd.DataFrame(results, columns=header)
    results_df.to_csv(VALIDATE_DIR.joinpath("results.csv"), index=False)

def get_results(baseline: pd.DataFrame, sr: pd.DataFrame) -> tuple:
    """
    Calculate summary metrics based on baseline and super-resolution (SR) DataFrames.
    
    Args:
        baseline (pd.DataFrame): DataFrame containing metrics calculated for baseline images.
        sr (pd.DataFrame): DataFrame containing metrics calculated for super-resolution images.

    Returns:
        tuple: A tuple containing three tuples for Correlation Coefficient, PSNR, and SSIM.
        Each inner tuple contains the count of valid values and the mean for baseline and SR images.
    """

    # Ensure both DataFrames have the same number of rows
    assert len(baseline) == len(sr), "Dataframes do not have the same length."

    # Initialize lists to store metrics
    baseline_corr_coeff, sr_corr_coeff = [], []
    baseline_psnr, sr_psnr = [], []
    baseline_ssim, sr_ssim = [], []
    
    # Iterate through rows of both DataFrames simultaneously
    for (row1, row2) in zip(baseline.itertuples(index=False), sr.itertuples(index=False)):
        # row = {index, boundary, corr_coeff, p_value, psnr, ssim}

        # Add Correlation Coefficient only if p_value is below 5%
        if not (math.isnan(row1[2]) or math.isnan(row2[2])) and (row1[3] < 0.05 or row2[3] < 0.05):
            baseline_corr_coeff.append(row1[2])
            sr_corr_coeff.append(row2[2])
        
        # Add PSNR values if they are not NaN
        if not (math.isnan(row1[4]) or math.isnan(row2[4])):
            baseline_psnr.append(row1[4])
            sr_psnr.append(row2[4])
        
        # Add SSIM values if they are not NaN
        if not (math.isnan(row1[5]) or math.isnan(row2[5])):
            baseline_ssim.append(row1[5])
            sr_ssim.append(row2[5])

    # Calculate summary statistics
    corr_coeff = (len(baseline_corr_coeff), np.mean(baseline_corr_coeff), np.mean(sr_corr_coeff))
    psnr = (len(baseline_psnr), np.mean(baseline_psnr), np.mean(sr_psnr))
    ssim = (len(baseline_ssim), np.mean(baseline_ssim), np.mean(sr_ssim))

    return corr_coeff, psnr, ssim

def visualize(model: str, s2_band: str, ps_band: str):
    """
    Visualize the performance metrics of a given model for specified bands over various months.
    
    Args:
        model (str): The name of the super-resolution model.
        s2_band (str): The Sentinel-2 band to consider.
        ps_band (str): The PlanetScope band to consider.
    """
    # Read the consolidated results into a DataFrame
    df = pd.read_csv(VALIDATE_DIR.joinpath("results.csv"))
    
    # Filter the DataFrame based on the selected Sentinel-2 and PlanetScope bands
    filtered_df = df.loc[(df["s2_band"] == s2_band) & (df["ps_band"] == ps_band)]
    
    # List of months to consider
    months = ['03_mar', '04_apr', '05_may', '06_jun', '07_jul', '08_aug', '09_sep']

    # Initialize lists to store metrics for each month
    corr_coeff, psnr, ssim = [], [], []
    baseline_corr_coeff, baseline_psnr, baseline_ssim = [], [], []

    # Populate the metric lists for each month
    for month in months:
        # Extract metrics for the current month and model and add them to the lists
        corr_coeff.append(filtered_df.loc[(filtered_df["month"] == month) & (filtered_df["model"] == model), "corr_coeff"].values[0])
        baseline_corr_coeff.append(filtered_df.loc[(filtered_df["month"] == month) & (filtered_df["model"] == model), "corr_coeff_baseline"].values[0])
        
        psnr.append(filtered_df.loc[(filtered_df["month"] == month) & (filtered_df["model"] == model), "psnr"].values[0])
        baseline_psnr.append(filtered_df.loc[(filtered_df["month"] == month) & (filtered_df["model"] == model), "psnr_baseline"].values[0])
        
        ssim.append(filtered_df.loc[(filtered_df["month"] == month) & (filtered_df["model"] == model), "ssim"].values[0])
        baseline_ssim.append(filtered_df.loc[(filtered_df["month"] == month) & (filtered_df["model"] == model), "ssim_baseline"].values[0])

    # Create output directory if it doesn't exist
    out_dir = VALIDATE_DIR.joinpath("visualize")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare to plot metrics
    months = ["March", "April", "May", "June", "July", "August", "September"]

    # Plotting Correlation Coefficient
    plot_metric(baseline_corr_coeff, corr_coeff, "Correlation Coefficient", model, months, s2_band, ps_band, out_dir, 0.7, 1.0)
    
    # Plotting PSNR
    plot_metric(baseline_psnr, psnr, "PSNR", model, months, s2_band, ps_band, out_dir, 20, 35)
    
    # Plotting SSIM
    plot_metric(baseline_ssim, ssim, "SSIM", model, months, s2_band, ps_band, out_dir, 0.7, 1.0)
    
def plot_metric(baseline, model_values, metric_name, model, months, s2_band, ps_band, out_dir, ymin, ymax):
    """
    Helper function to plot a specific metric.
    """
    out_name = f"{metric_name.lower()}_{s2_band}_{ps_band}_{model}"
    out_path = out_dir.joinpath(out_name)
    
    plt.plot(baseline, label='Bicubic', marker='o', linestyle='-', color='blue')
    plt.plot(model_values, label=model.upper(), marker='o', linestyle='-', color='red')

    plt.xlabel('Month')
    plt.ylabel(metric_name)
    plt.ylim(ymin, ymax)
    plt.title(f'{metric_name} for {s2_band} and {ps_band} - {model.upper()}')
    plt.xticks(np.arange(len(months)), months)
    plt.legend()

    plt.savefig(out_path.with_suffix(".png"))
    plt.close()


if __name__ == '__main__':
    create_results()