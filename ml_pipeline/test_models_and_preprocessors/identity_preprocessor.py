from interfaces.base_preprocessor import BasePreprocessor


class IdentityPreprocessor(BasePreprocessor):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def get_config(self):
        return {"type": "IdentityPreprocessor"}