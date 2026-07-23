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


def confusion_matrix(y_true, y_pred):
    """
    Compute the confusion matrix.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.

    Returns:
        np.ndarray: Confusion matrix.
    """
    unique_labels = np.unique(np.concatenate((y_true, y_pred)))
    num_classes = len(unique_labels)
    matrix = np.zeros((num_classes, num_classes), dtype=int)

    for true, pred in zip(y_true, y_pred):
        true_index = np.where(unique_labels == true)[0][0]
        pred_index = np.where(unique_labels == pred)[0][0]
        matrix[true_index, pred_index] += 1

    return matrix
