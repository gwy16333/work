import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import matplotlib.pyplot as plt

def load_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def get_dataloader(data_X, data_Y, batch_size, shuffle=True):
    tensor_X = torch.Tensor(data_X)
    tensor_Y = torch.Tensor(data_Y)
    dataset = TensorDataset(tensor_X, tensor_Y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def draw_gt_pred(gt, pred=None):
    """
    Draws the ground truth and predictions.

    Args:
        gt: ground truth data
        pred: predictions data (optional)

    Returns:
        None
    """
    plt.figure(figsize=(20, 8), dpi=120)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.tick_params(labelsize=18)
    plt.xlim(0, len(gt) + 4)
    plt.ylim(0, max(np.max(gt), np.max(pred)) + 50 if pred is not None else np.max(gt) + 50)

    # Plot ground truth
    plt.plot(gt, '-', color='blue', linewidth=1, label='Ground Truth')

    # Plot predictions only if they are provided
    if pred is not None:
        plt.plot(pred, '-.', color='orangered', linewidth=1.5, label='Predictions')

    plt.legend()
    plt.xlabel('Time Step', fontsize=18)
    plt.ylabel('Value', fontsize=18)
    plt.title('PEMS_04 Dataset 9 Day Ground Truth', fontsize=20)
    plt.show()

def evaluate_model(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return mae, mse, rmse, mape
