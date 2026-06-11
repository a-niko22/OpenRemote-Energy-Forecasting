# models/mean_model.py
from interfaces.base_model import BaseModel
import numpy as np


class MeanModel(BaseModel):
    """Predicts per-horizon mean of training targets."""

    def fit(self, X, y, X_val=None, y_val=None,
            epochs=1, batch_size=None, learning_rate=None, **kwargs):
        self.mean = np.mean(y, axis=0)
        return self

    def predict(self, X):
        return np.tile(self.mean, (len(X), 1))

    def get_config(self):
        return {"type": "MeanModel"}