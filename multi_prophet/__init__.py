from .prophet import Prophet
from .factories import model_pool_factory, dataframe_builder_factory

__version__ = "1.0.1"


class MultiProphet:

    def __init__(self, columns=[], config=None, regressors={}, **kwargs):
        self.model_pool = model_pool_factory(columns=columns,
                                             config=config,
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
        columns = self._columns(columns)

        for model in self._models(columns):
            model.add_seasonality(**kwargs)

    def add_country_holidays(self, country_name, columns=None):
        columns = self._columns(columns)

        for model in self._models(columns):
            model.add_country_holidays(country_name)

    def add_regressor(self, name, columns=None, **kwargs):
        columns = self._columns(columns)

        for model in self._models(columns):
            model.add_regressor(name, **kwargs)

        self._add_regressor_to_builder(name, columns)

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

    def cross_validation(self, horizon, **kwargs):
        return {
            column: model.cross_validation(horizon=horizon, **kwargs)
            for column, model in self.model_pool.items()
        }

    def performance_metrics(self, horizon, **kwargs):
        return {
            column: model.performance_metrics(horizon=horizon, **kwargs)
            for column, model in self.model_pool.items()
        }

    def _init_model_pool(self, columns, **kwargs):
        return {c: Prophet(**kwargs) for c in columns}

    def _first_model(self):
        return list(self.model_pool.values())[0]

    def _create_dataframe(self, df, column, train=False):
        return self.df_builder.create_df(df, column, train=train)

    def _contains_columns(self, df, column):
        return column in df.columns

    def _add_regressor_to_builder(self, name, columns):
        self.df_builder.add_regressor(name, columns)

    def _models(self, columns):
        return [self.model_pool[c] for c in columns]

    def _columns(self, columns):
        if columns:
            return columns
        else:
            return self.model_pool.keys()
