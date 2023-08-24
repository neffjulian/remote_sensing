from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

results_path = Path(__file__).parent.parent.parent.joinpath("data", "validate", "results.csv")
validate_path = Path(__file__).parent.parent.parent.joinpath("data", "validate")
processed_path = Path(__file__).parent.parent.parent.joinpath("data", "processed", "20m")

MONTH_MAP = {"03_mar": "Mar", "04_apr": "Apr", "05_may": "May", "06_jun": "Jun", "07_jul": "Jul", "08_aug": "Aug", "09_sep": "Sep"}

def eval_fields(sentinel: str):
    if sentinel == "10m":
        planet = "8b"
    else:
        planet = "4b"

    months = list(MONTH_MAP.keys())
    result = []
    for month in months:
        results = validate_path.joinpath(sentinel + "_rrdb", month, planet + "_results.csv")
        df = pd.read_csv(results)
        data = df.to_numpy()
        
        for entry in data:
            field_name = entry[0][:4] + "_" + entry[1][:3]
            result.append((field_name, entry[2], entry[4], entry[5]))
            
    unique_fields = set([x[0] for x in result])
    mean_result = []

    for field in unique_fields:
        field_result = [x for x in result if x[0] == field and x[3] >= 0]
        if len(field_result) == 7:
            corr_coeff = np.mean([x[1] for x in field_result])
            psnr = np.mean([x[2] for x in field_result])
            ssim = np.mean([x[3] for x in field_result])
            score = corr_coeff * psnr * ssim
            mean_result.append((field, corr_coeff, psnr, ssim, score))

    df = pd.DataFrame(mean_result, columns=["field_id", "corr_coeff", "psnr", "ssim", "score"])
    df.to_csv(validate_path.joinpath(sentinel + "_rrdb", planet + "_results.csv"), index=False)

# eval_fields("10m")

# def print_means(df: pd.DataFrame, s2: str, ps: str, model: str):
#     df_filtered = df[(df["s2_band"] == s2) & (df["ps_band"] == ps) & (df["model"] == model)]
#     data = df_filtered.to_numpy()
#     psnr, ssim, corr = [], [], []
#     for entry in data:
#         psnr.append(entry[9])
#         ssim.append(entry[12])
#         corr.append(entry[6])
#     print(s2 + " " + ps + " " + model + " (PSNR: " + str(np.mean(psnr)) + ", SSIM: " + str(np.mean(ssim)) + ", Corr: " + str(np.mean(corr)) + ")")

# def eval_bands(df: pd.DataFrame, model: str):
#     df_10m_4b = df[(df["s2_band"] == "10m") & (df["ps_band"] == "8b") & (df["model"] == model)]
#     # df_10m_4b.sort_values(by=["month"], inplace=True)
#     data_10m_4b = df_10m_4b.to_numpy()

#     df_20m_8b = df[(df["s2_band"] == "20m") & (df["ps_band"] == "4b") & (df["model"] == model)]
#     # df_20m_8b.sort_values(by=["month"], inplace=True)
#     data_20m_8b = df_20m_8b.to_numpy()
   
#     baseline_10m_4b = df[(df["s2_band"] == "10m") & (df["ps_band"] == "8b") & (df["model"] == "bicubic")]
#     # baseline_10m_4b.sort_values(by=["month"], inplace=True)
#     baseline_10m_4b = baseline_10m_4b.to_numpy()

#     baseline_20m_8b = df[(df["s2_band"] == "20m") & (df["ps_band"] == "4b") & (df["model"] == "bicubic")]
#     # baseline_20m_8b.sort_values(by=["month"], inplace=True)
#     baseline_20m_8b = baseline_20m_8b.to_numpy()

#     assert len(data_10m_4b) == len(data_20m_8b) == len(baseline_10m_4b) == len(baseline_20m_8b) == 7
#     print([MONTH_MAP[x] for x in data_10m_4b[:, 2]])
#     # Plot PSNR
#     plt.plot([MONTH_MAP[x] for x in data_10m_4b[:, 2]], data_10m_4b[:, 6], label="S2: 10m, PS: 8b", color="red")
#     plt.plot([MONTH_MAP[x] for x in data_20m_8b[:, 2]], data_20m_8b[:, 6], label="S2: 20m, PS: 4b", color="blue")
#     plt.plot([MONTH_MAP[x] for x in baseline_10m_4b[:, 2]], baseline_10m_4b[:, 6], color="red", linestyle="--")
#     plt.plot([MONTH_MAP[x] for x in baseline_20m_8b[:, 2]], baseline_20m_8b[:, 6], color="blue", linestyle="--")

#     plt.title("Correlation Coefficient")
#     plt.legend()
#     plt.show()

# df = pd.read_csv(results_path)
# eval_bands(df, "rrdb")
# for model in ["rrdb"]:
#     for s2 in ["20m", "10m"]:
#         for ps in ["4b", "8b"]:
#             print_means(df, s2, ps, model)




# def mean_and_std(files: list):
#     mean = []
#     std = []
#     for file in files:
#         data = np.load(file)
#         mean.append(np.mean(data))
#         std.append(np.std(data))
#     return np.mean(mean), np.mean(std)


# INDICES = ['0000', '0001', '0002', '0003', '0004', '0006', '0008', '0011', '0012', '0023', '0025', '0026', '0028', '0029', '0031', '0032', '0033', '0034', '0035', '0036', '0037', '0038', '0040', '0046']

# files = [processed for processed in processed_path.iterdir()]
# print(len(files))
# # files = [filename for filename in files if filename.name[3:7] not in INDICES]


# mar = [filename for filename in files if filename.name[:2] == "03"]
# mean, std = mean_and_std(mar)
# print("March:", mean, std)

# apr = [filename for filename in files if filename.name[:2] == "04"]
# mean, std = mean_and_std(apr)
# print("April:", mean, std)

# may = [filename for filename in files if filename.name[:2] == "05"]
# mean, std = mean_and_std(may)
# print("May:", mean, std)

# jun = [filename for filename in files if filename.name[:2] == "06"]
# mean, std = mean_and_std(jun)
# print("June:", mean, std)

# jul = [filename for filename in files if filename.name[:2] == "07"]
# mean, std = mean_and_std(jul)
# print("July:", mean, std)

# aug = [filename for filename in files if filename.name[:2] == "08"]
# mean, std = mean_and_std(aug)
# print("August:", mean, std)

# sep = [filename for filename in files if filename.name[:2] == "09"]
# mean, std = mean_and_std(sep)
# print("September:", mean, std)

# print_means(df, "10m", "4b", "rrdb")
# print(df.shape)
# df_filtered = df[(df["s2_band"] == "10m") & (df["ps_band"] == "4b") & (df["model"] == "rrdb")]
# df_filtered2 = df[(df["s2_band"] == "10m") & (df["ps_band"] == "8b") & (df["model"] == "rrdb")]


# data1 = df_filtered.to_numpy()
# data2 = df_filtered2.to_numpy()
# # SSIM: data[x][12]
# # PSNR: data[x][9]
# # Corr Coeff: data[x][6]
# print(data1[1][6])


# csv_path = Path(__file__).parent.parent.parent.joinpath("data", "results", "20m_ESRGAN_2023_08_21_11_17", "lai_preds.csv")

# df = pd.read_csv(csv_path, index_col=0)
# data = df.to_numpy()
# data = data[:-2]
# lai_lr = data[:, 0]
# lai_sr = data[:, 1]
# lai_hr = data[:, 2]
# lai_in_situ = data[:, 3]

# diff_lr = np.abs(lai_lr - lai_in_situ)
# diff_sr = np.abs(lai_sr - lai_in_situ)
# diff_hr = np.abs(lai_hr - lai_in_situ)

# l1_lr = np.mean(diff_lr)
# l1_sr = np.mean(diff_sr)
# l1_hr = np.mean(diff_hr)

# sq_lr = np.square(diff_lr)
# sq_sr = np.square(diff_sr)
# sq_hr = np.square(diff_hr)

# mean_lr = np.mean(sq_lr)
# mean_sr = np.mean(sq_sr)
# mean_hr = np.mean(sq_hr)

# root_lr = np.sqrt(mean_lr)
# root_sr = np.sqrt(mean_sr)
# root_hr = np.sqrt(mean_hr)

# r2_lr = r2_score(lai_in_situ, lai_lr)
# r2_sr = r2_score(lai_in_situ, lai_sr)
# r2_hr = r2_score(lai_in_situ, lai_hr)

# print("L1:", "LR:", l1_lr, "SR:", l1_sr, "HR:", l1_hr)
# print("RMSE:", "LR:", root_lr, "SR:", root_sr, "HR:", root_hr)
# print("R2:", "LR:", r2_lr, "SR:", r2_sr, "HR:", r2_hr)
