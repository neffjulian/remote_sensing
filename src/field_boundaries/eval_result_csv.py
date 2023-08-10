from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.metrics import r2_score

csv_path = Path(__file__).parent.parent.parent.joinpath("data", "results", "20m_RRDB_2023_08_10_08_35", "lai_preds.csv")

df = pd.read_csv(csv_path, index_col=0)
data = df.to_numpy()
data = data[:-2]
lai_lr = data[:, 0]
lai_sr = data[:, 1]
lai_hr = data[:, 2]
lai_in_situ = data[:, 3]

diff_lr = np.abs(lai_lr - lai_in_situ)
diff_sr = np.abs(lai_sr - lai_in_situ)
diff_hr = np.abs(lai_hr - lai_in_situ)

l1_lr = np.mean(diff_lr)
l1_sr = np.mean(diff_sr)
l1_hr = np.mean(diff_hr)

sq_lr = np.square(diff_lr)
sq_sr = np.square(diff_sr)
sq_hr = np.square(diff_hr)

mean_lr = np.mean(sq_lr)
mean_sr = np.mean(sq_sr)
mean_hr = np.mean(sq_hr)

root_lr = np.sqrt(mean_lr)
root_sr = np.sqrt(mean_sr)
root_hr = np.sqrt(mean_hr)

r2_lr = r2_score(lai_in_situ, lai_lr)
r2_sr = r2_score(lai_in_situ, lai_sr)
r2_hr = r2_score(lai_in_situ, lai_hr)

print("L1:", "LR:", l1_lr, "SR:", l1_sr, "HR:", l1_hr)
print("RMSE:", "LR:", root_lr, "SR:", root_sr, "HR:", root_hr)
print("R2:", "LR:", r2_lr, "SR:", r2_sr, "HR:", r2_hr)
