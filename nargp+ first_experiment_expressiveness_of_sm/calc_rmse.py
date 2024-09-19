import numpy as np

def calc_rmse(prediction, ground_truth):
    return np.sqrt(np.mean((prediction-ground_truth)**2))