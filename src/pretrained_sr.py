import numpy as np

def printarr(arr):
    print(f"[[{arr[0][0]}, {arr[0][1]}], [{arr[1][0]}, {arr[1][1]}]]")

arr = np.array([[1, 2], [3, 4]])
printarr(np.flip(arr, 1))
printarr(np.rot90(arr, 1))
printarr(np.rot90(arr, 2))
printarr(np.rot90(arr, 3))