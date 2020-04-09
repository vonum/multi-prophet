import unittest
import fbprophet
import multi_prophet

PREDICTOR_COLUMNS = ["y1", "y2"]

class FactoriesTestCase(unittest.TestCase):

    def test_no_args(self):
        model_pool = multi_prophet.model_pool_factory(columns=PREDICTOR_COLUMNS)

        self.assertEqual(PREDICTOR_COLUMNS, list(model_pool.keys()))

        y1_model = model_pool["y1"]
        self.assertIsInstance(y1_model, multi_prophet.Prophet)
        self.assertIsInstance(y1_model.prophet, fbprophet.Prophet)

        y2_model = model_pool["y2"]
        self.assertIsInstance(y2_model.prophet, fbprophet.Prophet)

    def test_equal_models_pool_factory(self):
        model_pool = multi_prophet.model_pool_factory(
            columns=PREDICTOR_COLUMNS,
            growth="logistic",
            weekly_seasonality=True,
            n_changepoints=50
        )

        for model in model_pool.values():
            self.assertEqual("logistic", model.prophet.growth)
            self.assertEqual(True, model.prophet.weekly_seasonality)
            self.assertEqual(50, model.prophet.n_changepoints)

    def test_different_models_pool_factory(self):
        args_dict = {
            "y1": {"growth": "logistic", "daily_seasonality": True},
            "y2": {"growth": "linear", "weekly_seasonality": True},
        }

        model_pool = multi_prophet.model_pool_factory(args_dict=args_dict)

        y1_model = model_pool["y1"]
        self.assertEqual("logistic", y1_model.prophet.growth)
        self.assertEqual(True, y1_model.prophet.daily_seasonality)

        y2_model = model_pool["y2"]
        self.assertEqual("linear", y2_model.prophet.growth)
        self.assertEqual(True, y2_model.prophet.weekly_seasonality)

    def test_invalid_kwargs(self):
        with self.assertRaises(TypeError):
            model_pool = multi_prophet.model_pool_factory(
                columns=PREDICTOR_COLUMNS,
                invalid="logistic"
            )
