import fbprophet
from . import plots


class Prophet:

    def __init__(self, **kwargs):
        self.prophet = fbprophet.Prophet(**kwargs)

    def fit(self, df, **kwargs):
        self.prophet.fit(df, **kwargs)

    def make_future_dataframe(self, periods):
        return self.prophet.make_future_dataframe(periods=periods)

    def predict(self, future_df):
        return self.prophet.predict(future_df)

    def add_seasonality(self, **kwargs):
        self.prophet.add_seasonality(**kwargs)

    def add_country_holidays(self, country_name):
        self.prophet.add_country_holidays(country_name=country_name)

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
