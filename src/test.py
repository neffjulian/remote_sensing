import numpy as np
import pandas as pd
from pathlib import Path

s2_dir = Path(__file__).parent.parent.joinpath('data', 'processed', '20m_in_situ')
ps_dir = Path(__file__).parent.parent.joinpath('data', 'processed', '4b_in_situ')

source_dir = Path(__file__).parent.parent.joinpath('data', 'coordinates', 'field_data.csv')
tar_dir = Path(__file__).parent.parent.joinpath('data', 'coordinates', 'lai_est.csv')


def main():
    # Note: My indices start at 0 and the field ids start at 1.
    field_data = pd.read_csv(source_dir, sep=',')
    lai = field_data['lai'].to_numpy()

    error_s2 = []
    s2_lai = []
    indices = []
    for i, file in enumerate(s2_dir.iterdir()):
        field_id = int(file.name[0:4]) + 1
        data = np.load(file)
        s2_lai.append(data[12, 12])
        error_s2.append(abs(lai[field_id - 1] - data[12, 12]))
        indices.append(int(file.name[0:4]) + 1)

    error_ps = []
    ps_lai = []
    for i, file in enumerate(ps_dir.iterdir()):
        field_id = int(file.name[0:4]) + 1
        data = np.load(file)
        ps_lai.append((data[74, 75] + data[75, 74] + data[74, 74] + data[75, 75]) / 4)
        error_ps.append(abs(lai[field_id - 1] - (data[74, 75] + data[75, 74] + data[74, 74] + data[75, 75]) / 4))

    lai_ = [lai[index - 1] for index in indices]

    for i in range(len(lai_)):
        print(indices[i], lai_[i], s2_lai[i], ps_lai[i], error_s2[i], error_ps[i])

    df = pd.DataFrame({'field_id': indices, 'lai_in_situ': lai_, 'lai_est_s2': s2_lai, 'lai_est_ps': ps_lai, 'error_s2': error_s2, 'error_ps': error_ps})
    df.to_csv(tar_dir, index=False)

    print(df['error_s2'].mean(), df['error_ps'].mean())

if __name__ == '__main__':
    main()