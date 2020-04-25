import unittest
import numpy as np
import pandas as pd
from multi_prophet import data_builder

PREDICTOR_COLUMNS = ["y", "y1"]


class DataBuilderTestCase(unittest.TestCase):

    def setUp(self):
        self.df = pd.read_csv("tests/data/example_wp_log_peyton_manning.csv")
        self.df["y1"] = self.df["y"]

    def test_creating_training_df_basic(self):
        df_builder = data_builder.DataFrameBuilder({})

        y_df = df_builder.create_df(self.df, "y", train=True)
        np.testing.assert_array_equal(["ds", "y"], y_df.columns)

        y1_df = df_builder.create_df(self.df, "y1", train=True)
        np.testing.assert_array_equal(["ds", "y"], y1_df.columns)

    def test_capacity_floor_training_df(self):
        self.df["floor_y"] = 50
        self.df["cap_y"] = 500
        self.df["cap_y1"] = 500

        df_builder = data_builder.DataFrameBuilder({})

        y_df = df_builder.create_df(self.df, "y", train=True)
        np.testing.assert_array_equal(["ds", "y", "cap", "floor"], y_df.columns)

        y1_df = df_builder.create_df(self.df, "y1", train=True)
        np.testing.assert_array_equal(["ds", "y", "cap"], y1_df.columns)

    def test_regressors_training_df(self):
        regressors = {"y": ["y1"]}
        df_builder = data_builder.DataFrameBuilder(regressors)

        y_df = df_builder.create_df(self.df, "y", train=True)
        np.testing.assert_array_equal(["ds", "y", "y1"], y_df.columns)

        y1_df = df_builder.create_df(self.df, "y1", train=True)
        np.testing.assert_array_equal(["ds", "y"], y1_df.columns)

    def test_creating_prediction_df_basic(self):
        df_builder = data_builder.DataFrameBuilder({})

        y_df = df_builder.create_df(self.df, "y")
        np.testing.assert_array_equal(["ds"], y_df.columns)

        y1_df = df_builder.create_df(self.df, "y1")
        np.testing.assert_array_equal(["ds"], y1_df.columns)

    def test_capacity_floor_prediction_df(self):
        self.df["floor_y"] = 50
        self.df["cap_y"] = 500
        self.df["cap_y1"] = 500

        df_builder = data_builder.DataFrameBuilder({})

        y_df = df_builder.create_df(self.df, "y")
        np.testing.assert_array_equal(["ds", "cap", "floor"], y_df.columns)

        y1_df = df_builder.create_df(self.df, "y1")
        np.testing.assert_array_equal(["ds", "cap"], y1_df.columns)

    def test_regressors_prediction_df(self):
        regressors = {"y": ["y1"]}
        df_builder = data_builder.DataFrameBuilder(regressors)

        y_df = df_builder.create_df(self.df, "y")
        np.testing.assert_array_equal(["ds", "y1"], y_df.columns)

        y1_df = df_builder.create_df(self.df, "y1")
        np.testing.assert_array_equal(["ds"], y1_df.columns)
