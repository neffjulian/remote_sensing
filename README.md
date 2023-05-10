# Remote-Sensing

## Project Structure

```
├── remote_sensing
│   ├── data
|   |   ├── coordinates <- Contains geographic coordinates from different locations in Switzerland
|   |   ├── filtered <- Contains intermediate data where outliers have been removed (by hand usually)
|   |   ├── processed <- Contains the final, canonical data sets for modeling
|   |   ├── raw <- Contains the original, immutable data dump
│   ├── src
|   |   ├── download 
|   |   |   ├── eschikon_download.py <- Needed for downloading validation data
|   |   |   ├── planetscope_download.py <- Downloads PlanetScope data
|   |   |   ├── sentinel_download.py <- Downloads Sentinel-2 data
|   |   |   ├── utils.py <- Utility function for the files above
│   ├── .gitignore <- Which files to ignore (e.g. data)
|   |   ├── model 
|   |   |   ├── cnn.py <- Implementation of the CNN model
|   |   ├── preprocess 
|   |   |   ├── histogram_matching.py <- Implementation of the histogram matching.
|   |   |   ├── planetscope_preprocess.py <- Preprocesses Planetscope files (filtered->processed)
|   |   |   ├── sentinel_preprocess.py <- Preprocesses Sentinel-2 files (filtered->processed)
|   |   |   ├── utils.py <- Utility function for the files above
│   ├── .env <- Used to hide confidential data
│   ├── environment.yml <- Contains all dependencies for an easy setup
│   ├── README.md <- The top-level README for developers using this project
```

## Downloading Images
- Go to "/src"
- python download_images.py --satellite sentinel --year 2022 --month mar

