from bdpy.evals.metrics import profile_correlation, pattern_correlation, pairwise_identification


import pickle
import unittest

import numpy as np


class TestEval(unittest.TestCase):
    def test_profile_correlation(self):
        # 2-d array
        n = 30
        x = np.random.rand(10, n)
        y = np.random.rand(10, n)
        r = np.array([[
            np.corrcoef(x[:, i], y[:, i])[0, 1]
            for i in range(n)
        ]])

        self.assertTrue(np.array_equal(
            profile_correlation(x, y), r
        ))
        self.assertEqual(profile_correlation(x, y).shape, (1, n))

        # Multi-d array
        x = np.random.rand(10, 4, 3, 2)
        y = np.random.rand(10, 4, 3, 2)
        xf = x.reshape(10, -1)
        yf = y.reshape(10, -1)
        r = np.array([[
            np.corrcoef(xf[:, i], yf[:, i])[0, 1]
            for i in range(4 * 3 * 2)
        ]])
        r = r.reshape(1, 4, 3, 2)

        self.assertTrue(np.array_equal(
            profile_correlation(x, y), r
        ))
        self.assertEqual(profile_correlation(x, y).shape, (1, 4, 3, 2))

    def test_pattern_correlation(self):
        # 2-d array
        x = np.random.rand(10, 30)
        y = np.random.rand(10, 30)
        r = np.array([
            np.corrcoef(x[i, :], y[i, :])[0, 1]
            for i in range(10)
        ])

        self.assertTrue(np.array_equal(
            pattern_correlation(x, y), r
        ))
        self.assertEqual(pattern_correlation(x, y).shape, (10,))

        # Multi-d array
        x = np.random.rand(10, 4, 3, 2)
        y = np.random.rand(10, 4, 3, 2)
        xf = x.reshape(10, -1)
        yf = y.reshape(10, -1)
        r = np.array([
            np.corrcoef(xf[i, :], yf[i, :])[0, 1]
            for i in range(10)
        ])

        self.assertTrue(np.array_equal(
            pattern_correlation(x, y), r
        ))
        self.assertEqual(pattern_correlation(x, y).shape, (10,))

    def test_2d(self):
        with open('data/testdata-2d.pkl.gz', 'rb') as f:
            d = pickle.load(f)
        self.assertTrue(np.array_equal(
            profile_correlation(d['x'], d['y']),
            d['r_prof']
        ))
        self.assertTrue(np.array_equal(
            pattern_correlation(d['x'], d['y']),
            d['r_patt']
        ))
        self.assertTrue(np.array_equal(
            pairwise_identification(d['x'], d['y']),
            d['ident_acc']
        ))

    def test_2d_nan(self):
        with open('data/testdata-2d-nan.pkl.gz', 'rb') as f:
            d = pickle.load(f)
        # self.assertTrue(np.array_equal(
        #     profile_correlation(d['x'], d['y']),
        #     d['r_prof']
        # ))
        self.assertTrue(np.array_equal(
            pattern_correlation(d['x'], d['y'], remove_nan=True),
            d['r_patt'],
        ))
        self.assertTrue(np.array_equal(
            pairwise_identification(d['x'], d['y'], remove_nan=True),
            d['ident_acc'],
        ))

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEval)
    unittest.TextTestRunner(verbosity=2).run(suite)
