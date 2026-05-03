from interfaces.code.base_preprocessor import BasePreprocessor
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from Preprocessors.PreprocessorsObjects import CyclicalScalling

class StandardPreprocessor(BasePreprocessor):
    """
    This preprocessor is intended to be used as the standard for Exp2. It applies quite standard and simple preprocessing to float and datetime features.
    The methods are mapped to the datatypes. Therefore, we can always add new datatypes and methods to it if needed.

    datetime* has an * on it to mark the way it should be treated. 

    If there are any questions about this. Feel free to ask me (Gui). If you'd like for the mapper to be dynamic, please ask me to implement it.
    """
    _methodMapperPerType = {
        "float" : [StandardScaler(), MinMaxScaler()],
        "datetime*" : [CyclicalScalling()]
    }

    def transform(self, X : DataFrame) -> DataFrame:
        all_columns = X.columns.to_series()
        for dtype, methods in self._methodMapperPerType.items():
            columns_of_dtype = all_columns[X.dtypes == dtype]
            for method in methods:
                if dtype.__contains__("*"):
                    X = method.fit_transform(X)
                else:
                    X[columns_of_dtype] = method.fit_transform(X[columns_of_dtype])
        return X

    def fit(self, X : DataFrame, y = None):
        pass