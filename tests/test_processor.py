import unittest
import pandas as pd
import numpy as np
import os
import sys

# Add the project root to sys.path to import the package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fico_bucketing.processor import FICOQuantizer, BucketingStrategyOptimizer

class TestFICOQuantizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a sample dataset for testing
        data = {
            'fico_score': [400, 450, 500, 550, 600, 650, 700, 750, 800, 850],
            'default': [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        }
        cls.df = pd.DataFrame(data)
        cls.quantizer = FICOQuantizer(cls.df)

    def test_init_validation(self):
        # Test missing columns
        bad_df = pd.DataFrame({'wrong_col': [1, 2]})
        with self.assertRaises(ValueError):
            FICOQuantizer(bad_df)

    def test_equal_width_buckets(self):
        boundaries = self.quantizer.equal_width_buckets(n_buckets=2)
        self.assertEqual(len(boundaries), 3)
        self.assertEqual(boundaries[0], 400)
        self.assertEqual(boundaries[-1], 850)

    def test_equal_frequency_buckets(self):
        boundaries = self.quantizer.equal_frequency_buckets(n_buckets=2)
        self.assertEqual(len(boundaries), 3)
        # Median of [400...850] is 625
        self.assertAlmostEqual(boundaries[1], 625.0)

    def test_calculate_iv(self):
        boundaries = [400, 600, 850]
        iv = self.quantizer._calculate_iv(boundaries)
        self.assertGreater(iv, 0)

    def test_optimize_information_value(self):
        best_boundaries = self.quantizer.optimize_information_value(n_buckets=2)
        self.assertEqual(len(best_boundaries), 3)
        # Best boundary should separate defaults (1) from non-defaults (0)
        # With the simple synthetic dataset, it should be between 500 and 650
        self.assertTrue(500 <= best_boundaries[1] <= 650)

class TestBucketingStrategyOptimizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set seed for repeatability
        np.random.seed(42)
        data = {
            'fico_score': np.random.randint(400, 850, 1000),
            'default': np.random.randint(0, 2, 1000)
        }
        cls.df = pd.DataFrame(data)
        cls.quantizer = FICOQuantizer(cls.df)
        cls.optimizer = BucketingStrategyOptimizer(cls.quantizer)

    def test_compare_strategies(self):
        comparison_df = self.optimizer.compare_strategies(bucket_ranges=[5, 10])
        self.assertFalse(comparison_df.empty)
        self.assertIn('method', comparison_df.columns)
        self.assertIn('information_value', comparison_df.columns)

    def test_get_recommended_strategy(self):
        comparison_df = self.optimizer.compare_strategies(bucket_ranges=[5])
        rec = self.optimizer.get_recommended_strategy(comparison_df)
        self.assertIn('recommended_method', rec)
        self.assertIn('recommended_n_buckets', rec)

if __name__ == '__main__':
    unittest.main()
