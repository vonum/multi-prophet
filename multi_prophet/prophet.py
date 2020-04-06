import fbprophet


class Prophet:

    def __init__(self):
        self.prophet = fbprophet.Prophet()

    def fit(self, df):
        self.prophet.fit(df)

    def make_future_dataframe(self, periods):
        return self.prophet.make_future_dataframe(periods=periods)

    def predict(self, future_df):
        return self.prophet.predict(future_df)
