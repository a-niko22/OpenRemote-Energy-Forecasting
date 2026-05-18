from interfaces.base_preprocessor import BasePreprocessor
from interfaces.base_model import BaseModel


class Experiment:
    def __init__(self, name, preprocessor, model, input_len=None, horizon=None):
        if not isinstance(preprocessor, BasePreprocessor):
            raise TypeError("preprocessor must inherit from BasePreprocessor.")

        if not isinstance(model, BaseModel):
            raise TypeError("model must inherit from BaseModel.")

        self.name = name
        self.preprocessor = preprocessor
        self.model = model
        self.input_len = input_len
        self.horizon = horizon
