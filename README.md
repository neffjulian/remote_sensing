# Remote-Sensing

## Project Structure

```
├── remote_sensing
│   ├── configs <- Consists of different configuration files for each model.
│   ├── data
│   │   ├── coordinates <- Contains coordinate files of the data and the LAI predictions.
│   │   ├── filtered <- Contains ordered raw data where unusable (too noisy/little data available) outliers have been removed (by hand usually).
│   │   ├── processed <- Created after preprocessing. Contains preprocessed data divided by band.
│   │   ├── raw <- Contains the original, immutable data dump. Not included in the download script.
│   │   ├── results <- Contains the results of the models.
│   │   ├── validate <- Contains the data needed to validate the field boundaries.
│   ├── src
│   │   ├── field_boundaries <- Scripts related to the creation and validation of field boundaries.
│   │   │   ├── create_boundaries.py <- Creates the field boundaries using PlanetScope data and official parcel boundaries provided by [GeoDienste.ch](https://www.geodienste.ch/services/lwb_nutzungsflaechen).
│   │   │   ├── prepare_validation_data.py <- Uses a pretrained model and band selection of Sentinel-2 data to create Super Resolved Sentinel-2 boundaries.
│   │   │   ├── validate_boundaries.py <- Compares the boundaries created and stores the results in a CSV.
│   │   │   ├── visualize_results.py <- Used for analyzing the different CSVs in validate_boundaries.py
│   │   ├── model <- Contains model implementations and utility functions for various Super-Resolution models.
│   │   │   ├── dataset.py <- Pytorch Lightning Datamodule class implementation of the dataset used for this project.
│   │   │   ├── edsr.py <- Pytorch Lightning implementation of EDSR.
│   │   │   ├── esrgan.py <- Pytorch Lightning implementation of ESRGAN.
│   │   │   ├── rrdb.py <- Pytorch Lightning implementation of RRDB.
│   │   │   ├── srcnn.py <- Pytorch Lightning implementation of SRCNN.
│   │   │   ├── srdiff_no_pos_encoding.py <- Pytorch Lightning implementation of SRDIFF with no positional encodings.
│   │   │   ├── srdiff.py <- Pytorch Lightning implementation of SRDIFF.
│   │   │   ├── srgan.py <- Pytorch Lightning implementation of SRGAN.
│   │   │   ├── srresnet.py <- Pytorch Lightning implementation of SRResNet.
│   │   │   ├── utils.py <- Collection of utility functions used for Super-Resolution.
│   │   ├── satellites <- Scripts related to satellite data handling.
│   │   │   ├── create_in_situ_geojson.py <- Creates a geojson using in-situ data points
│   │   │   ├── planetscope_download.py <- Used to download PlanetScope data.
│   │   │   ├── sentinel_download.py <- Used to download Sentinel-2 data.
│   │   │   ├── utils.py <- Collection of utility functions used for satellite data handling.
│   │   ├── download_data.py <- Script to download the data used for training the models and the pretrained models originating from this work.
│   │   ├── main.py <- Used for training different models. Need to have downloaded and preprocessed the data first.
│   │   ├── preprocess_data.py <- Preprocesses data from 'filtered' and stores it in 'processed'.
│   ├── weights <- Stores the model weights, including SRCNN, EDSR, RRDB, downloadable via the download_data.py script.
│   ├── .env <- Used to store the Planet API Key.
```


## Getting Started

### Install Dependencies

#### Using pip

1. Make sure you have Python 3.x and pip installed on your system.
2. Navigate to the project's root directory.
3. Install the required dependencies using the following command:

```shell
pip install -r requirements.txt
```

#### Using conda
1. Make sure you have Anaconda or Miniconda installed on your system.
2. Navigate to the project's root directory.
3. Create a new conda environment using the following command:

```shell
conda env create -f environment.yml

```

#### API Key
Make sure to place your Planet API Key in the `.env` file before running the scripts.

#### Download Data

1. Run `download_data.py` script to download data used for training. The data has already been filtered and georeferenced by hand.

```shell
python src/download/download_data.py

```

#### Preprocess Data

1. Run `preprocess_data.py` script to preprocess data from the `filtered` directory.¨

```shell
python src/preprocess_data.py

```

#### Run the Model

1. Run `main.py` script with your favorite model.

```shell
python src/model/main.py

```

Note: Depending on the specific environment and configuration, the commands might vary slightly. Please adjust the commands accordingly to fit your system setup.



