#! /bin/bash

if ! command -v conda >/dev/null 2>&1; then
    echo "Conda is not installed. Aborting script."
    exit 1
fi

conda create -n remote_sensing
source activate remote_sensing

echo "Create a new conda environment "remote_sensing."
conda create -n remote_sensing

cd ../src/download


# wget
# py7zr
# opencv-python
# eodal
# scikit-image
# numpy
# torch
# pytorch-lightning
# lightning[extra]

https://polybox.ethz.ch/index.php/s/AdiQZeI9YGmp9Ac