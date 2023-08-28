from pathlib import Path
import pandas as pd
import numpy as np

out = Path(__file__).parent.parent.joinpath("data", "validate", "in_situ_eval.csv")

rrdb_10_path = Path(__file__).parent.parent.joinpath("data", "results", "10m_RRDB_2023_08_21_09_41", "lai_preds.csv")
edsr_10_path = Path(__file__).parent.parent.joinpath("data", "results", "10m_EDSR_2023_08_21_09_32", "lai_preds.csv")
esrgan_10_path = Path(__file__).parent.parent.joinpath("data", "results", "10m_ESRGAN_2023_08_21_09_59", "lai_preds.csv")
srcnn_10_path = Path(__file__).parent.parent.joinpath("data", "results", "10m_SRCNN_2023_08_21_09_23", "lai_preds.csv")
edsr_20_path = Path(__file__).parent.parent.joinpath("data", "results", "20m_EDSR_2023_08_21_10_48", "lai_preds.csv")
esrgan_20_path = Path(__file__).parent.parent.joinpath("data", "results", "20m_ESRGAN_2023_08_21_11_17", "lai_preds.csv")
srcnn_20_path = Path(__file__).parent.parent.joinpath("data", "results", "20m_SRCNN_2023_08_21_10_41", "lai_preds.csv")
rrdb_20_path = Path(__file__).parent.parent.joinpath("data", "results", "20m_RRDB_2023_08_21_11_00", "lai_preds.csv")

models = (("RRDB_10", pd.read_csv(rrdb_10_path)),
            ("EDSR_10", pd.read_csv(edsr_10_path)),
            ("ESRGAN_10", pd.read_csv(esrgan_10_path)),
            ("SRCNN_10", pd.read_csv(srcnn_10_path)),
            ("EDSR_20", pd.read_csv(edsr_20_path)),
            ("ESRGAN_20", pd.read_csv(esrgan_20_path)),
            ("SRCNN_20", pd.read_csv(srcnn_20_path)),
            ("RRDB_20", pd.read_csv(rrdb_20_path))
)

def calculate_errors(list: np.ndarray, month: str, name: str):
    s2_error, sr_error, hr_error = [], [], []
    model, res = name.split("_")

    for pred in list:
        s2_error.append(abs(pred[1] - pred[4]))
        sr_error.append(abs(pred[2] - pred[4]))
        hr_error.append(abs(pred[3] - pred[4]))

    # Calculate mean absolute error
    s2_mae = np.mean(s2_error)
    sr_mae = np.mean(sr_error)
    hr_mae = np.mean(hr_error)

    # Calculate root mean squared error
    s2_rmse = np.sqrt(np.mean(np.square(s2_error)))
    sr_rmse = np.sqrt(np.mean(np.square(sr_error)))
    hr_rmse = np.sqrt(np.mean(np.square(hr_error)))

    # # Calculate R2 score
    in_situ = [pred[4] for pred in list]
    s2 = [pred[1] for pred in list]
    sr = [pred[2] for pred in list]
    hr = [pred[3] for pred in list]

    s2_r2 = np.corrcoef(in_situ, s2)[0,1] ** 2
    sr_r2 = np.corrcoef(in_situ, sr)[0,1] ** 2
    hr_r2 = np.corrcoef(in_situ, hr)[0,1] ** 2

    # Write to csv
    with open(out, "a") as f:
        f.write(f"{model},{res},{month},{s2_mae},{sr_mae},{hr_mae},{s2_rmse},{sr_rmse},{hr_rmse},{s2_r2},{sr_r2},{hr_r2}\n")
        f.close()

def eval(name, preds):
    print("Eval: ", name)
    # to numpy
    preds = preds.to_numpy()

    # pred = [index, s2_lai, sr_lai, hr_lai, in_situ_lai, date]
    mar, apr, may, jun = [], [], [], []
    for pred in preds:
        if int(pred[5][5:7]) == 3:
            mar.append(pred)
        elif int(pred[5][5:7]) == 4:
            apr.append(pred)
        elif int(pred[5][5:7]) == 5:
            may.append(pred)
        elif int(pred[5][5:7]) == 6:
            jun.append(pred)
        else:
            print("Error")

    print(len(mar), len(apr), len(may), len(jun))
    calculate_errors(np.array(preds), "All months", name)
    calculate_errors(np.array(mar), "March", name)
    calculate_errors(np.array(apr), "April", name)
    calculate_errors(np.array(may), "May", name)
    calculate_errors(np.array(jun), "June", name)


def main():
    # Create eval csv and add header
    with open(out, "w") as f:
        f.write("model,resolution,month,s2_mae,sr_mae,hr_mae,s2_rmse,sr_rmse,hr_rmse,s2_r2,sr_r2,hr_r2\n")
        # f.write("")
        f.close()
    
    for name, preds in models:
        eval(name, preds)

if __name__ == "__main__":
    main()