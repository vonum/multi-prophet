import pandas as pd
from .prophet import Prophet

__version__ = "0.1"
TIME_COLUMN = "ds"


class MultiProphet:

    def __init__(self, columns, **kwargs):
        self.model_pool = self._init_model_pool(columns, **kwargs)

    def fit(self, df, **kwargs):
        for column, model in self.model_pool.items():
            mdf = self._create_dataframe(df, column)
            model.fit(mdf, **kwargs)

    def make_future_dataframe(self, periods):
        model = self._first_model()
        return model.make_future_dataframe(periods)

    def predict(self, future_df):
        return {
            column: model.predict(future_df)
            for column, model in self.model_pool.items()
        }

    def add_seasonality(self, **kwargs):
        for model in self.model_pool.values():
            model.add_seasonality(**kwargs)

    def add_country_holidays(self, country_name):
        for model in self.model_pool.values():
            model.add_country_holidays(country_name)

    def _init_model_pool(self, columns, **kwargs):
        return {c: Prophet(**kwargs) for c in columns}

    def _first_model(self):
        return list(self.model_pool.values())[0]

    def _create_dataframe(self, df, column):
        return pd.DataFrame({
          "ds": df[TIME_COLUMN].values,
          "y": df[column].values
        })
