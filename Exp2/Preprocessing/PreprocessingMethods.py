from interfaces.code.base_preprocessor import BasePreprocessor
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from Preprocessors.PreprocessorsObjects import CyclicalScalling
from feature_engine.datetime import DatetimeFeatures
from pywt import wavedec, waverec
import numpy as np
from sklearn.preprocessing import StandardScaler
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

class DWTPreprocessor(BasePreprocessor):
    """
    This preprocessor applies the Discrete Wavelet Transform to the data. It is intended to be used as a baseline for the DWT method in Exp2.b.
    """
    _dimensionGroups= [['temperature_2m', 'cloud_cover', 'wind_speed_10m', 'shortwave_radiation'], ['total_load', 'generation_forecast', 'Price', 'High', 'Low', 'Change %', 'price']]
    def transform(self, X : DataFrame) -> DataFrame:
        dtf = DatetimeFeatures(features_to_extract=["hour", "day_of_week", "month", "quarter", "weekend"], drop_original=False)

        df = dtf.fit_transform(X)

        for group in self._dimensionGroups:
            for col in group:
                if col in df.columns:
                    data = df[col].values.copy()
                    coeffs = wavedec(data, 'sym4', 'symmetric', level=3)

                    scaler = StandardScaler()

                    for i in range(1, len(coeffs)):
                        detail_coeffs = [np.zeros_like(c) for c in coeffs]
                        detail_coeffs[i] = coeffs[i]
                        df[f"{col}_d{i}"] = waverec(detail_coeffs, 'sym4')[:len(df)]
                        df[f"{col}_d{i}"] = scaler.fit_transform(df[[f"{col}_d{i}"]])

        return df
    def fit(self, X : DataFrame, y = None):
        pass