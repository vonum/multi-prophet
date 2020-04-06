import pdb
import unittest
import numpy as np
import pandas as pd
import fbprophet
import multi_prophet


class MultiProphetTestCase(unittest.TestCase):

    def setUp(self):
        self.df = pd.read_csv("tests/data/example_wp_log_peyton_manning.csv")

    def test_version(self):
        self.assertEqual("0.1", multi_prophet.__version__)

    def test_constructor(self):
        mp = multi_prophet.Prophet()
        self.assertIsNotNone(mp)
        self.assertIsInstance(mp.prophet, fbprophet.Prophet)

    def test_fit(self):
        mp = multi_prophet.Prophet()
        mp.fit(self.df)
        # will be tested later by checking the performance of the model
        self.assertTrue(True)

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
