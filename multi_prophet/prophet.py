import prophet
from prophet.diagnostics import cross_validation, performance_metrics
from . import plots


class Prophet:
    def __init__(self, **kwargs):
        self.prophet = prophet.Prophet(**kwargs)

    def fit(self, df, **kwargs):
        self.prophet.fit(df, **kwargs)

    def make_future_dataframe(self, periods, **kwargs):
        return self.prophet.make_future_dataframe(periods=periods, **kwargs)

    def predict(self, future_df):
        return self.prophet.predict(future_df)

    def add_seasonality(self, **kwargs):
        self.prophet.add_seasonality(**kwargs)

    def add_country_holidays(self, country_name):
        self.prophet.add_country_holidays(country_name=country_name)

    def add_regressor(self, name, **kwargs):
        self.prophet.add_regressor(name, **kwargs)

    def plot(self, forecast, plotly=False, **kwargs):
        if plotly:
            return plots.plotly_plot(self.prophet, forecast, **kwargs)
        else:
            return self.prophet.plot(forecast)

    def plot_components(self, forecast, plotly=False, **kwargs):
        if plotly:
            return plots.plotly_components_plot(self.prophet, forecast, **kwargs)
        else:
            return self.prophet.plot_components(forecast)

    def cross_validation(self, horizon, **kwargs):
        return cross_validation(self.prophet, horizon=horizon, **kwargs)

    def performance_metrics(self, horizon, **kwargs):
        cv_df = cross_validation(self.prophet, horizon=horizon, **kwargs)
        return performance_metrics(cv_df)
