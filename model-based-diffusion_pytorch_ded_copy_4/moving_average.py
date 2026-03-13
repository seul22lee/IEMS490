import numpy as np

def moving_average_1d(data, window_size):
    if window_size < 1:
        raise ValueError("Window size must be at least 1")
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode='valid')