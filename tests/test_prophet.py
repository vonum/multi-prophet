import unittest
import numpy as np
import pandas as pd
import matplotlib
import plotly
import prophet
import multi_prophet


class ProphetTestCase(unittest.TestCase):
    def setUp(self):
        self.df = pd.read_csv("tests/data/example_wp_log_peyton_manning.csv")

    def test_constructor(self):
        mp = multi_prophet.Prophet()
        self.assertIsNotNone(mp)
        self.assertIsInstance(mp.prophet, prophet.Prophet)

    def test_constructor_kwargs(self):
        mp = multi_prophet.Prophet(growth="logistic")
        self.assertEqual("logistic", mp.prophet.growth)

    def test_constructor_invalid_kwargs(self):
        with self.assertRaises(TypeError):
            mp = multi_prophet.Prophet(invalid="logistic")

    def test_fit(self):
        mp = multi_prophet.Prophet()
        mp.fit(self.df)
        # will be tested later by checking the performance of the model
        self.assertTrue(True)

    def test_fit_logistic(self):
        mp = multi_prophet.Prophet(growth="logistic")
        self.df["cap"] = 200
        self.df["floor"] = 0
        mp.fit(self.df)
        # will be tested later by checking the performance of the model
        self.assertTrue(True)

    def test_fit_kwargs(self):
        mp = multi_prophet.Prophet()
        mp.fit(self.df, algorithm="Newton")
        # will be tested later by checking the performance of the model
        self.assertTrue(True)

    def test_fit_invalid_kwargs(self):
        mp = multi_prophet.Prophet()
        with self.assertRaises(ValueError):
            mp.fit(self.df, invalid="Newton")

    def test_make_future_dataframe_length(self):
        mp = multi_prophet.Prophet()
        mp.fit(self.df)
        future_df = mp.make_future_dataframe(7)

        initial_len = len(self.df)

        self.assertEqual(initial_len + 7, len(future_df))

    def test_make_future_dataframe_dates(self):
        dates = [
            "2016-01-21", "2016-01-22", "2016-01-23", "2016-01-24",
            "2016-01-25", "2016-01-26", "2016-01-27"
        ]

        mp = multi_prophet.Prophet()
        mp.fit(self.df)
        future_df = mp.make_future_dataframe(7)

        future_dates = [
            np.datetime_as_string(d, unit="D")
            for d in future_df.tail(7)["ds"].values
        ]

        self.assertEqual(dates, future_dates)

    def test_predict(self):
        columns = [
            "ds", "trend", "yhat_lower", "yhat_upper",
            "trend_lower", "trend_upper", "additive_terms", "additive_terms_lower",
            "additive_terms_upper", "weekly", "weekly_lower", "weekly_upper",
            "yearly", "yearly_lower", "yearly_upper", "multiplicative_terms",
            "multiplicative_terms_lower", "multiplicative_terms_upper", "yhat"
        ]

        mp = multi_prophet.Prophet()
        mp.fit(self.df)
        future_df = mp.make_future_dataframe(7)

        forecast = mp.predict(future_df)

        self.assertEqual(len(future_df), len(forecast))
        self.assertEqual(columns, list(forecast.columns.values))

    def test_add_seasonality(self):
        mp = multi_prophet.Prophet()
        mp.add_seasonality(name="monthly", period=30.5, fourier_order=5)
        seasonality = mp.prophet.seasonalities["monthly"]

        self.assertEqual(30.5, seasonality["period"])
        self.assertEqual(5, seasonality["fourier_order"])
        self.assertEqual(10, seasonality["prior_scale"])
        self.assertEqual("additive", seasonality["mode"])
        self.assertIsNone(seasonality["condition_name"])

    def test_add_country_holiday(self):
        mp = multi_prophet.Prophet()
        mp.add_country_holidays(country_name="US")

        mp.fit(self.df)
        future_df = mp.make_future_dataframe(7)

        self.assertEqual(14, len(mp.prophet.train_holiday_names))

    def test_add_regressor(self):
        mp = multi_prophet.Prophet()
        mp.add_regressor("Matchday")

        extra_regressor = mp.prophet.extra_regressors["Matchday"]
        self.assertEqual(10.0, extra_regressor["prior_scale"])
        self.assertEqual("auto", extra_regressor["standardize"])
        self.assertEqual(0.0, extra_regressor["mu"])
        self.assertEqual(1.0, extra_regressor["std"])
        self.assertEqual("additive", extra_regressor["mode"])

    def test_plot(self):
        mp = multi_prophet.Prophet()
        mp.fit(self.df)

        future_df = mp.make_future_dataframe(7)
        forecast = mp.predict(future_df)

        self.assertIsInstance(mp.plot(forecast), matplotlib.figure.Figure)

    def test_plotly_plot(self):
        mp = multi_prophet.Prophet()
        mp.fit(self.df)

        future_df = mp.make_future_dataframe(7)
        forecast = mp.predict(future_df)

        self.assertIsInstance(mp.plot(forecast, plotly=True), plotly.graph_objs.Figure)

    def test_components_plot(self):
        mp = multi_prophet.Prophet()
        mp.fit(self.df)

        future_df = mp.make_future_dataframe(7)
        forecast = mp.predict(future_df)

        self.assertIsInstance(
            mp.plot_components(forecast),
            matplotlib.figure.Figure
        )

    def test_plotly_components_plot(self):
        mp = multi_prophet.Prophet()
        mp.fit(self.df)

        future_df = mp.make_future_dataframe(7)
        forecast = mp.predict(future_df)

        self.assertIsInstance(
            mp.plot_components(forecast, plotly=True),
            plotly.graph_objs.Figure
        )

    def test_cross_validation(self):
        mp = multi_prophet.Prophet()
        mp.fit(self.df)

        cross_validation_df = mp.cross_validation(horizon="365 days")
        self.assertIsInstance(cross_validation_df, pd.DataFrame)

        self.assertTrue("ds" in cross_validation_df.columns)
        self.assertTrue("yhat" in cross_validation_df.columns)
        self.assertTrue("yhat_lower" in cross_validation_df.columns)
        self.assertTrue("yhat_upper" in cross_validation_df.columns)
        self.assertTrue("y" in cross_validation_df.columns)
        self.assertTrue("cutoff" in cross_validation_df.columns)

    def test_performance_metrics(self):
        mp = multi_prophet.Prophet()
        mp.fit(self.df)

        performance_metrics_df = mp.performance_metrics(horizon="365 days")
        self.assertIsInstance(performance_metrics_df, pd.DataFrame)

        self.assertTrue("horizon" in performance_metrics_df.columns)
        self.assertTrue("mse" in performance_metrics_df.columns)
        self.assertTrue("rmse" in performance_metrics_df.columns)
        self.assertTrue("mae" in performance_metrics_df.columns)
        self.assertTrue("mape" in performance_metrics_df.columns)
        self.assertTrue("mdape" in performance_metrics_df.columns)
        self.assertTrue("coverage" in performance_metrics_df.columns)
