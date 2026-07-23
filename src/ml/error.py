import numpy as np


def MSE(y_true, y_pred):
    """
    Mean Squared Error (MSE) loss function.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.

    Returns:
        float: Mean squared error.
    """
    return np.mean((y_true - y_pred) ** 2)
