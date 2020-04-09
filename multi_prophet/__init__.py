import pandas as pd
from .prophet import Prophet
from .factories import model_pool_factory

__version__ = "0.1"
TIME_COLUMN = "ds"


class MultiProphet:

    def __init__(self, columns=[], args_dict=None, **kwargs):
        self.model_pool = model_pool_factory(columns=columns,
                                             args_dict=args_dict,
                                             **kwargs)

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

    def add_regressor(self, name, **kwargs):
        for model in self.model_pool.values():
            model.add_regressor(name, **kwargs)

    def plot(self, forecasts, plotly=False, **kwargs):
        return {
            column: self.model_pool[column].plot(forecast, plotly=plotly, **kwargs)
            for column, forecast in forecasts.items()
        }

    def plot_components(self, forecasts, plotly=False, **kwargs):
        return {
            column: self.model_pool[column].plot_components(forecast,
                                                            plotly=plotly,
                                                            **kwargs)
            for column, forecast in forecasts.items()
        }

    def _init_model_pool(self, columns, **kwargs):
        return {c: Prophet(**kwargs) for c in columns}

    def _first_model(self):
        return list(self.model_pool.values())[0]

    def _create_dataframe(self, df, column):
        return pd.DataFrame({
          "ds": df[TIME_COLUMN].values,
          "y": df[column].values
        })
