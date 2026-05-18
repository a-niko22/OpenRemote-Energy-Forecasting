from interfaces.base_preprocessor import BasePreprocessor
import numpy as np


class MinMaxPreprocessor(BasePreprocessor):
    def fit(self, X, y=None):
        X = np.asarray(X)
        flat = X.reshape(-1, X.shape[-1])
        self.mins = flat.min(axis=0)
        self.maxs = flat.max(axis=0)
        self.span = np.where(self.maxs - self.mins == 0, 1, self.maxs - self.mins)
        return self

    def transform(self, X):
        return (np.asarray(X) - self.mins) / self.span

    def get_config(self):
        return {"type": "MinMaxPreprocessor"}