# models/zero_model.py
from interfaces.base_model import BaseModel
import numpy as np


class ZeroModel(BaseModel):
    """Predicts 0 for every input."""

    def fit(self, X, y, X_val=None, y_val=None,
            epochs=1, batch_size=None, learning_rate=None, **kwargs):
        self.horizon = y.shape[1] if y.ndim > 1 else 1
        return self

    def predict(self, X):
        return np.zeros((len(X), self.horizon))

    def get_config(self):
        return {"type": "ZeroModel"}