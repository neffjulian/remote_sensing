"""
Used for evaluating the CSV which is created after super-resolution on Sentinel-2 data. 

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

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

results_path = Path(__file__).parent.parent.parent.joinpath("data", "validate", "results.csv")
validate_path = Path(__file__).parent.parent.parent.joinpath("data", "validate")
processed_path = Path(__file__).parent.parent.parent.joinpath("data", "processed", "20m")

MONTH_MAP = {"03_mar": "Mar", "04_apr": "Apr", "05_may": "May", "06_jun": "Jun", "07_jul": "Jul", "08_aug": "Aug", "09_sep": "Sep"}

def eval_fields(sentinel: str) -> None:
    """Evaluate fields to compute various quality metrics using the specified satellite data resolution.

    Args:
        sentinel (str): The spatial resolution of the Sentinel-2 data to be used. E.g. "10m", "20m".

    Outputs:
        A CSV file containing aggregated quality metrics is saved in the specified directory.
    """

    # Assign planet bands based on sentinel bands
    if sentinel == "10m":
        planet = "8b"
    else:
        planet = "4b"

    # Get list of months from the MONTH_MAP keys
    months = list(MONTH_MAP.keys())
    result = []

    # Iterate through each month
    for month in months:
        # Construct the path to the CSV file for that month
        results = validate_path.joinpath(sentinel + "_rrdb", month, planet + "_results.csv")

        # Read the CSV into a DataFrame
        df = pd.read_csv(results)
        data = df.to_numpy()

        # Create a list of tuples with relevant data from each row
        for entry in data:
            field_name = entry[0][:4] + "_" + entry[1][:3]
            result.append((field_name, entry[2], entry[4], entry[5]))

    # Create a set of unique field names
    unique_fields = set([x[0] for x in result])
    mean_result = []

    # Calculate the mean metrics for each unique field
    for field in unique_fields:
        field_result = [x for x in result if x[0] == field and x[3] >= 0]

        # Consider only fields with exactly 7 entries
        if len(field_result) == 7:
            corr_coeff = np.mean([x[1] for x in field_result])
            psnr = np.mean([x[2] for x in field_result])
            ssim = np.mean([x[3] for x in field_result])
            score = corr_coeff * psnr * ssim

            mean_result.append((field, corr_coeff, psnr, ssim, score))

    # Create a DataFrame and save it as a CSV
    df = pd.DataFrame(mean_result, columns=["field_id", "corr_coeff", "psnr", "ssim", "score"])
    df.to_csv(validate_path.joinpath(sentinel + "_rrdb", planet + "_results.csv"), index=False)

def print_means(df: pd.DataFrame, s2: str, ps: str, model: str) -> None:
    """Calculate and print the mean PSNR, SSIM, and Correlation Coefficient for a specific satellite band and super-resolution model.

    Args:
        df (pd.DataFrame): The DataFrame containing metrics and model information.
        s2 (str): The Sentinel-2 band specification. E.g., "10m", "20m".
        ps (str): The PlanetScope band specification. E.g., "4b", "8b".
        model (str): The name of the super-resolution model to evaluate. E.g., "rrdb", "srcnn".

    Output:
        Prints the mean values of PSNR, SSIM, and Correlation Coefficient for the specified conditions.
    """

    # Filter the DataFrame based on the specified satellite bands and super-resolution model
    df_filtered = df[(df["s2_band"] == s2) & (df["ps_band"] == ps) & (df["model"] == model)]

    # Convert the filtered DataFrame to a NumPy array
    data = df_filtered.to_numpy()

    # Initialize lists to hold the values of PSNR, SSIM, and Correlation Coefficient
    psnr, ssim, corr = [], [], []

    # Populate the lists with corresponding values from the data array
    for entry in data:
        psnr.append(entry[9])
        ssim.append(entry[12])
        corr.append(entry[6])

    # Calculate and print the mean values
    print(f"{s2} {ps} {model} (PSNR: {np.mean(psnr):.2f}, SSIM: {np.mean(ssim):.2f}, Corr: {np.mean(corr):.2f})")

def eval_bands(df: pd.DataFrame, model: str) -> None:
    """Evaluate and plot the correlation coefficient for satellite bands using different super-resolution models.

    Args:
        df (pd.DataFrame): The DataFrame containing columns 's2_band', 'ps_band', 'model', and correlation coefficients.
        model (str): The name of the super-resolution model to evaluate. E.g. "rrdb", "srcnn".
    """
    
    # Filter data for 10m and 8b bands for the specified model
    df_10m_4b = df[(df["s2_band"] == "10m") & (df["ps_band"] == "8b") & (df["model"] == model)]
    data_10m_4b = df_10m_4b.to_numpy()
    
    # Filter data for 20m and 4b bands for the specified model
    df_20m_8b = df[(df["s2_band"] == "20m") & (df["ps_band"] == "4b") & (df["model"] == model)]
    data_20m_8b = df_20m_8b.to_numpy()

    # Filter data for 10m and 8b bands for the baseline model (bicubic)
    baseline_10m_4b = df[(df["s2_band"] == "10m") & (df["ps_band"] == "8b") & (df["model"] == "bicubic")]
    baseline_10m_4b = baseline_10m_4b.to_numpy()

    # Filter data for 20m and 4b bands for the baseline model (bicubic)
    baseline_20m_8b = df[(df["s2_band"] == "20m") & (df["ps_band"] == "4b") & (df["model"] == "bicubic")]
    baseline_20m_8b = baseline_20m_8b.to_numpy()

    # Ensure all filtered datasets have the same length
    assert len(data_10m_4b) == len(data_20m_8b) == len(baseline_10m_4b) == len(baseline_20m_8b) == 7

    # Debug print to show months considered in the plot
    print([MONTH_MAP[x] for x in data_10m_4b[:, 2]])

    # Plot the correlation coefficients
    plt.plot([MONTH_MAP[x] for x in data_10m_4b[:, 2]], data_10m_4b[:, 6], label="S2: 10m, PS: 8b", color="red")
    plt.plot([MONTH_MAP[x] for x in data_20m_8b[:, 2]], data_20m_8b[:, 6], label="S2: 20m, PS: 4b", color="blue")
    plt.plot([MONTH_MAP[x] for x in baseline_10m_4b[:, 2]], baseline_10m_4b[:, 6], color="red", linestyle="--")
    plt.plot([MONTH_MAP[x] for x in baseline_20m_8b[:, 2]], baseline_20m_8b[:, 6], color="blue", linestyle="--")

    # Add title and legend to the plot
    plt.title("Correlation Coefficient")
    plt.legend()

    # Show the plot
    plt.show()