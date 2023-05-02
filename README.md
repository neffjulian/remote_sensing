# Remote-Sensing

## Project Structure

```
├── remote_sensing
│   ├── data
|   |   ├── coordinates <- Contains geographic coordinates from different locations in Switzerland
|   |   ├── interim <- Contains intermediate data that has been transformed
|   |   ├── processed <- Contains the final, canonical data sets for modeling
|   |   ├── raw <- Contains the original, immutable data dump
│   ├── src
|   |   ├── dataset 
|   |   |   ├── download_dataset.py <- Contains scripts to download the dataset or access dataset from the data folder
│   ├── .gitignore <- Tells Git which files to ignore when committing your project to the GitHub repository
│   ├── .env <- Used to hide confidential data
│   ├── environment.yml <- Contains all dependencies for an easy setup
│   ├── README.md <- The top-level README for developers using this project
```

## Downloading Images
- Go to "/src"
- python download_images.py --satellite sentinel --year 2022 --month mar

