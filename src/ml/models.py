import numpy as np
import pandas as pd

# from sklearn.linear_model import LogisticRegression
# def mse()


# ((X[i] * W + B) - y)^2


class LinearRegression:
    def __init__(self):

        self.W = None
        self.B: float = 0.0

    def fit(self, X: pd.DataFrame, Y: pd.Series, epochs: int, lr: float):

        x = X.to_numpy()

        y = Y.to_numpy().reshape(-1)

        if x.ndim == 1:
            x = x.reshape(-1, 1)

        datalen, num_features = x.shape

        self.W = np.zeros(num_features)

        for epoch in range(epochs):
            losses = x @ self.W + self.B - y  # (num_features, 1)

            dl_dw = (x.T @ losses) / datalen

            dl_db = np.sum(losses) / datalen

            self.W -= lr * dl_dw
            self.B -= lr * dl_db

            print(f"Epoch {epoch + 1}/{epochs} with loss {(losses**2).mean()}")

    def predict(self, X: pd.DataFrame):

        if self.W is None:
            raise ValueError(
                "Model is not trained yet. Please call fit() before predict()."
            )

        x = X.to_numpy()

        if x.ndim == 1:
            x = x.reshape(-1, 1)

        if x.shape[1] != self.W.shape[0]:
            raise ValueError(f"Expected {self.W.shape[0]} features, got {x.shape[1]}")

        if x.ndim == 1:
            x = x.reshape(-1, 1)

        return x @ self.W + self.B


class LogisticRegression:
    def __init__(self):
        self.W = None
        self.B = 0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X: pd.Series | pd.DataFrame, Y: pd.Series, epochs: int, lr: float):

        x = X.to_numpy()
        y = Y.to_numpy().reshape(-1)

        if x.ndim == 1:
            x = x.reshape(-1, 1)

        n, d = x.shape

        self.W = np.zeros(d)

        for epoch in range(epochs):
            z = x @ self.W + self.B
            p = self.sigmoid(z)

            residuals = p - y

            dw = (x.T @ residuals) / n
            db = np.sum(residuals) / n

            self.W -= lr * dw
            self.B -= lr * db

            loss = -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))

            print(f"Epoch {epoch + 1}/{epochs} loss={loss}")
