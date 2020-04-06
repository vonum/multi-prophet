import unittest
import multi_prophet


class MultiProphetTestCase(unittest.TestCase):

    def test_version(self):
        self.assertEqual("0.1", multi_prophet.__version__)

    def test_constructor(self):
        self.assertIsNotNone(multi_prophet.Prophet())
