# Guided Superresolution for Sentinel-2 and PlanetScope Satellite Data

## Getting Started

### Installation
To use this code, you need to have Python 3 installed on your computer. Then, you can install the necessary Python packages by running the following command in your terminal or command prompt:

```
pip install -r requirements.txt
```

This command will install all the dependencies required to run the code.

### Downloading Satellite Data
To download satellite images, run the following command in your terminal or command prompt:
```
python download_images.py --satellite <satellite-name> --year <year> --month <month>

```
Replace <satellite-name> with either "planetscope" or "sentinel-2", depending on which satellite you want to download images from. Replace <year> with the year you want the images to be taken from, and replace <month> with the month during which the images should have been taken.

For example, to download Sentinel-2 images taken in March 2022, run:

```
python download_images.py --satellite sentinel-2 --year 2022 --month march
```

The downloaded images will be stored in the data directory.

That's it! With these instructions, you should be able to set up and use the guided superresolution code.