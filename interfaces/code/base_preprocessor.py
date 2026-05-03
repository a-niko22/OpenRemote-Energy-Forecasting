from abc import ABC, abstractmethod


class BasePreprocessor(ABC):
    """Contract every preprocessor must fulfill to run in the pipeline.

    Preprocessors transform windowed data before it reaches the model.
    They may learn statistics from training data (fit) and apply them to
    train/val/test (transform).
    """

    @abstractmethod
    def fit(self, X, y=None):
        """Learn transformation parameters from training data.

        Args:
            X: training inputs, shape (n_windows, input_len, n_features)
            y: training targets, optional — needed by target-aware preprocessors

        Returns:
            self
        """
        pass

    @abstractmethod
    def transform(self, X, **kwargs):
        """Apply the learned transformation. Shape is preserved."""
        pass

    def inverse_transform(self, X):
        """Reverse the transformation. Optional — only needed if predictions
        must be inverted back to the original space (e.g. target scalers).
        Default raises NotImplementedError."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support inverse_transform."
        )

    def get_config(self):
        """Return a dict of parameters used, for logging/reproducibility."""
        return {}