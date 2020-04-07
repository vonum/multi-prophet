import fbprophet


class Prophet:

    def __init__(self, **kwargs):
        self.prophet = fbprophet.Prophet(**kwargs)

    def fit(self, df, **kwargs):
        self.prophet.fit(df, **kwargs)

    def make_future_dataframe(self, periods):
        return self.prophet.make_future_dataframe(periods=periods)

    def predict(self, future_df):
        return self.prophet.predict(future_df)
