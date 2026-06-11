from pandas import DataFrame
from interfaces.code.base_preprocessor import BasePreprocessor
from feature_engine.datetime import DatetimeFeatures
from sklearn.preprocessing import FunctionTransformer
import numpy as np
from pandas.api.types import is_datetime64_any_dtype

class CyclicalScalling(BasePreprocessor):
    """
    This class implements onto the DataFrame that it is given, cyclical scalling on all DateTime columns. The idea is that, because we only have 1 column for the timestamp, the class looks for it, regardless of name
    and applies cyclical scalling, decomposing that feature into more features, and getting rid of the datetime column altogher.

    Returns:
        - The DataFrame without the datetime column, and with all the new cyclical features :)
    """
    def transform(self, X : DataFrame, drop : bool = True) -> DataFrame:
        # Feel free to modify this mapping. Perhaps if you intend on using this class but want different mappings, ask me (Gui) to implement a way to make it dynamic.
        features_to_extract_period_mapping = {
            "hour" : 24,
            "day_of_week" : 7,
            "month" : 12,
            "quarter" : 4,
            "weekend" : 2,
        }
        features_to_extract = list(features_to_extract_period_mapping.keys())
        try:
            datetime_columns = [col for col in X.columns if is_datetime64_any_dtype(X[col])]
            dtf = DatetimeFeatures(features_to_extract=features_to_extract, drop_original=True)

            X["day"] = X[datetime_columns[0]].dt.day # Days are a wee bit different, since they vary depending on the month and year
            X = dtf.fit_transform(X)

            columns_with_prefix = [col for col in X.columns if col.startswith(datetime_columns[0])]
            for col in columns_with_prefix:
                X.rename(columns={col : col.replace(datetime_columns[0] + "_", "")}, inplace=True)

            for feature, period in features_to_extract_period_mapping.items():
                X[f"{feature}_sin"] = self.__sinTransformer(X[feature], period)
                X[f"{feature}_cos"] = self.__cosTransformer(X[feature], period)
                if drop:
                    X.drop(feature, axis=1, inplace=True)

            return X
        except ValueError:
            raise ValueError("The DataFrame does not contain any DateTime column, or the DateTime column is not in the correct format. Please make sure to have a DateTime column in the correct format before using this class.")
            
    def fit(self, X : DataFrame, y = None):
        pass

    def fit_transform(self, X : DataFrame, y = None) -> DataFrame:
        return self.transform(X)

    def __sinTransformer(self, datetime_series : DataFrame, period : int):
        return FunctionTransformer(lambda x: np.sin(2 * np.pi * x / period)).fit_transform(datetime_series) 
    
    def __cosTransformer(self, datetime_series : DataFrame, period : int):
        return FunctionTransformer(lambda x: np.cos(2 * np.pi * x / period)).fit_transform(datetime_series)
    
