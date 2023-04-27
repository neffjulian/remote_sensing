import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

def load_data(folder, filename):
    data = np.load(f'../../images/sentinel_2/{folder}/{filename}')
    return data

def iterate_files(folder):
    for filename in os.listdir(f'../../images/sentinel_2/{folder}'):
        if filename.endswith('.npz'):
            file = load_data(folder, filename)
            # TODO

def interpolate(arr, method = 'linear', target_shape = (400, 400)):
    """
    Perform interpolation on a 2D numpy array.

    Parameters
    ----------
    arr : numpy.ndarray
        Input 2D array to be interpolated.
    methid: str
        Method of interpolation to perform ("linear", "cubic", "nearest", etc...)
    target_shape : tuple, optional
        Shape of the output array. Default is (400, 400).

    Returns
    -------
    numpy.ndarray
        Interpolated 2D array.
    """

    new_arr = np.squeeze(arr)
    source_shape = np.shape(new_arr)
    x_old = np.linspace(0, 1, source_shape[0])
    x_new = np.linspace(0, 1, target_shape[0])

    new_arr /= new_arr.max()

    interp_func = RegularGridInterpolator(
        points = (x_old, x_old), 
        values = new_arr, 
        method = method)
    xx, yy = np.meshgrid(x_new, x_new, indexing='ij')
    points = np.stack((xx.ravel(), yy.ravel()), axis=-1)

    new_arr = interp_func(points).reshape(target_shape)
    return new_arr

def show(arr):
    fig, ax = plt.subplots()
    im = ax.imshow(arr)
    plt.show()

def show_rgb(data):
    blue = interpolate(data['B02'])
    green = interpolate(data['B03'])
    red = interpolate(data['B04'])

    new_arr = np.dstack((red, green, blue))
    show(new_arr)

if __name__ == "__main__":
    data = load_data("22_march", "0000.npz")
    show_rgb(data)