from .prophet import Prophet
from .factories import model_pool_factory, dataframe_builder_factory

__version__ = "0.2.0"


class MultiProphet:

    def __init__(self, columns=[], args_dict=None, regressors={}, **kwargs):
        self.model_pool = model_pool_factory(columns=columns,
                                             args_dict=args_dict,
                                             regressors=regressors,
                                             **kwargs)
        self.df_builder = dataframe_builder_factory(regressors)

    def fit(self, df, **kwargs):
        for column, model in self.model_pool.items():
            mdf = self._create_dataframe(df, column, train=True)
            model.fit(mdf, **kwargs)

    def make_future_dataframe(self, periods, **kwargs):
        model = self._first_model()
        return model.make_future_dataframe(periods, **kwargs)

    def predict(self, future_df):
        return {
            column: model.predict(self._create_dataframe(future_df, column))
            for column, model in self.model_pool.items()
        }

    def add_seasonality(self, columns=None, **kwargs):
        for model in self._columns_models(columns):
            model.add_seasonality(**kwargs)

    def add_country_holidays(self, country_name, columns=None):
        for model in self._columns_models(columns):
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

    def _create_dataframe(self, df, column, train=False):
        return self.df_builder.create_df(df, column, train=train)

    def _contains_columns(self, df, column):
        return column in df.columns

    def _columns_models(self, columns):
        if columns:
            return [self.model_pool[c] for c in columns]
        else:
            return self.model_pool.values()
