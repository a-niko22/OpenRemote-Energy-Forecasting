from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Contract every model must fulfill to run in the pipeline.

    fit() signature is deliberately flexible — NN models will use the extra args,
    baselines will ignore them via **kwargs.
    """

    @abstractmethod
    def fit(self, X, y, X_val=None, y_val=None,
            epochs=1, batch_size=None, learning_rate=None, **kwargs):
        """Train the model.

        Args:
            X:              training inputs, shape (n_windows, input_len, n_features)
            y:              training targets, shape (n_windows, horizon)
            X_val, y_val:   optional validation data for early stopping / monitoring
            epochs:         for iterative models (NNs). Ignored by one-shot models.
            batch_size:     for mini-batch training. Ignored by one-shot models.
            learning_rate:  optimizer learning rate. Ignored by non-gradient models.
            **kwargs:       any model-specific hyperparameters.

        Returns:
            self
        """
        pass

    @abstractmethod
    def predict(self, X):
        """Returns predictions of shape (n_windows, horizon)."""
        pass

    def get_config(self):
        """Return a dict of hyperparameters used, for logging/reproducibility.
        Default implementation returns an empty dict; models should override."""
        return {}