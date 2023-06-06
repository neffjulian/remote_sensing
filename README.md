# Remote-Sensing

## Project Structure

```
├── remote_sensing
│   ├── data
|   |   ├── coordinates 
|   |   |   ├── field_parcels.geojson <-  Locations of field parcels with available in-situ LAI measurements.
|   |   |   ├── points_ch.geojson <-  Locations of field parcels in Switzerland used for training.
|   |   ├── filtered <- Contains intermediate data where outliers have been removed (by hand usually)
|   |   ├── processed <- Contains the processed data used for training
|   |   ├── raw <- Contains the original, immutable data dump
│   ├── src
|   |   ├── download 
|   |   |   ├── download_data.py <- Script to download data used for training. The data has already been filtered and georeferenced by hand.
|   |   |   ├── planetscope_download.py <- Either places an order or downloads data. Uses the locations in 'coordinates/points.ch'
|   |   |   ├── sentinel_download.py <- Downloads Sentinel-2 data. Uses the locations in 'coordinates/points.ch'
|   |   |   ├── utils.py <- Utility function for the files above
|   |   ├── preprocess_data.py <- Preprocesses data from 'filtered' and stores it in 'processed'.
|   |   ├── model 
|   |   |   ├── dataset.py <- 
|   |   |   ├── edsr.py <- 
|   |   |   ├── srcnn.py <- 
|   |   |   ├── utils.py <- 
|   |   |   ├── main.py <- 
│   ├── .env <- Used to store the Planet API Key
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
Make sure to set up the Planet API Key in the `.env` file before running the scripts.

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



