import unittest
import numpy as np
import pandas as pd
import matplotlib
import multi_prophet

PREDICTOR_COLUMNS = ["y", "y1"]

class MultiProphetTestCase(unittest.TestCase):

    def setUp(self):
        self.df = pd.read_csv("tests/data/example_wp_log_peyton_manning.csv")
        self.df["y1"] = self.df["y"]

    def test_version(self):
        self.assertEqual("0.2.0", multi_prophet.__version__)

    def test_constructor(self):
        mp = multi_prophet.MultiProphet(columns=PREDICTOR_COLUMNS)
        self.assertIsNotNone(mp)

    def test_make_future_dataframe_length(self):
        mp = multi_prophet.MultiProphet(columns=PREDICTOR_COLUMNS)
        mp.fit(self.df)
        future_df = mp.make_future_dataframe(7)

        initial_len = len(self.df)

        self.assertEqual(initial_len + 7, len(future_df))

    def test_make_future_dataframe_dates(self):
        dates = [
            "2016-01-21", "2016-01-22", "2016-01-23", "2016-01-24",
            "2016-01-25", "2016-01-26", "2016-01-27"
        ]

        mp = multi_prophet.MultiProphet(columns=PREDICTOR_COLUMNS)
        mp.fit(self.df)
        future_df = mp.make_future_dataframe(7)

        future_dates = [
            np.datetime_as_string(d, unit="D")
            for d in future_df.tail(7)["ds"].values
        ]

        self.assertEqual(dates, future_dates)

    def test_predict(self):
        mp = multi_prophet.MultiProphet(columns=PREDICTOR_COLUMNS)
        mp.fit(self.df)
        future_df = mp.make_future_dataframe(7)

        forecast = mp.predict(future_df)

        self.assertEqual(2, len(forecast))
        self.assertIsInstance(forecast["y"], pd.DataFrame)
        self.assertIsInstance(forecast["y1"], pd.DataFrame)

    def test_add_seasonality(self):
        mp = multi_prophet.MultiProphet(columns=PREDICTOR_COLUMNS)
        mp.add_seasonality(name="monthly", period=30.5, fourier_order=5)

        for model in mp.model_pool.values():
            seasonality = model.prophet.seasonalities["monthly"]

            self.assertEqual(30.5, seasonality["period"])
            self.assertEqual(5, seasonality["fourier_order"])
            self.assertEqual(10, seasonality["prior_scale"])
            self.assertEqual("additive", seasonality["mode"])
            self.assertIsNone(seasonality["condition_name"])

    def test_add_country_holiday(self):
        mp = multi_prophet.MultiProphet(columns=PREDICTOR_COLUMNS)
        mp.add_country_holidays(country_name="US")

        mp.fit(self.df)
        future_df = mp.make_future_dataframe(7)

        for model in mp.model_pool.values():
            self.assertEqual(14, len(model.prophet.train_holiday_names))

    def test_add_regressor(self):
        mp = multi_prophet.MultiProphet(columns=PREDICTOR_COLUMNS)
        mp.add_regressor("Matchday")

        for model in mp.model_pool.values():
            extra_regressor = model.prophet.extra_regressors["Matchday"]
            self.assertEqual(10.0, extra_regressor["prior_scale"])
            self.assertEqual("auto", extra_regressor["standardize"])
            self.assertEqual(0.0, extra_regressor["mu"])
            self.assertEqual(1.0, extra_regressor["std"])
            self.assertEqual("additive", extra_regressor["mode"])

    def test_plot(self):
        mp = multi_prophet.MultiProphet(columns=PREDICTOR_COLUMNS)
        mp.fit(self.df)

        future_df = mp.make_future_dataframe(7)
        forecasts = mp.predict(future_df)

        plots = mp.plot(forecasts).values()

        for plot in plots:
          self.assertIsInstance(plot, matplotlib.figure.Figure)

    def test_components_plot(self):
        mp = multi_prophet.MultiProphet(columns=PREDICTOR_COLUMNS)
        mp.fit(self.df)

        future_df = mp.make_future_dataframe(7)
        forecasts = mp.predict(future_df)

        plots = mp.plot_components(forecasts).values()

        for plot in plots:
          self.assertIsInstance(plot, matplotlib.figure.Figure)
