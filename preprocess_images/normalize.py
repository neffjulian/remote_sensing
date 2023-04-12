import numpy as np

def normalize(arr):
    # Calculate the mean and standard deviation of the array
    mean = np.mean(arr)
    std = np.std(arr)
    
    # Normalize the array by subtracting the mean and dividing by the standard deviation
    normalized = (arr - mean) / std
    
    return normalized