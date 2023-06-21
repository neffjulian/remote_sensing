from pathlib import Path
import numpy as np

PROCESSED_DIR = Path(__file__).parent.parent.joinpath("data", "processed")

if __name__ == "__main__":
    s2_dir = PROCESSED_DIR.joinpath("20m")
    ps_dir = PROCESSED_DIR.joinpath("hist_20m_4b")

    files = [file.name for file in s2_dir.iterdir()]
    assert files == [file.name for file in ps_dir.iterdir()]

    errors = []
    for file in files:
        s2_file = np.load(s2_dir.joinpath(file))
        ps_file = np.load(ps_dir.joinpath(file))
        diff = np.abs(s2_file - ps_file)
        errors.append((diff, file))

    sorted_errors = sorted(errors, key=lambda x: x[0])

    print(sorted_errors[0:10])