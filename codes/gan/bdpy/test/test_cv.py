# coding: utf-8
'''Tests for ml'''


import os
import unittest
import shutil
import pickle

import numpy as np

from bdpy.ml.crossvalidation import cvindex_groupwise, make_cvindex

from sklearn.linear_model import LinearRegression
from fastl2lir import FastL2LiR


class TestCv(unittest.TestCase):

    def test_cvindex_groupwise(self):

        # Test data
        x = np.array([
            1, 1, 1,
            2, 2, 2,
            3, 3, 3,
            4, 4, 4,
            5, 5, 5,
            6, 6, 6
        ])

        # Expected output
        train_index = [
            np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]),
            np.array([0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]),
            np.array([0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17]),
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17]),
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17]),
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
        ]

        test_index = [
            np.array([ 0,  1,  2]),
            np.array([ 3,  4,  5]),
            np.array([ 6,  7,  8]),
            np.array([ 9, 10, 11]),
            np.array([12, 13, 14]),
            np.array([15, 16, 17])
        ]

        cvindex = cvindex_groupwise(x)

        for i, (tr, te) in enumerate(cvindex):
            self.assertTrue(np.array_equal(train_index[i], tr))
            self.assertTrue(np.array_equal(test_index[i], te))

    def test_cvindex_groupwise_exclusive(self):

        # Test data
        x = np.array([
            1, 1, 1,
            2, 2, 2,
            3, 3, 3,
            4, 4, 4,
            5, 5, 5,
            6, 6, 6
        ])

        # Exclusive labels
        a = np.array([
            1, 2, 3,
            4, 5, 6,
            1, 2, 3,
            4, 5, 6,
            1, 2, 3,
            4, 5, 6,
        ])

        # Expected output
        train_index = [
            np.array([3, 4, 5, 9, 10, 11, 15, 16, 17]),
            np.array([0, 1, 2, 6,  7,  8, 12, 13, 14]),
            np.array([3, 4, 5, 9, 10, 11, 15, 16, 17]),
            np.array([0, 1, 2, 6,  7,  8, 12, 13, 14]),
            np.array([3, 4, 5, 9, 10, 11, 15, 16, 17]),
            np.array([0, 1, 2, 6,  7,  8, 12, 13, 14])
        ]

        test_index = [
            np.array([ 0,  1,  2]),
            np.array([ 3,  4,  5]),
            np.array([ 6,  7,  8]),
            np.array([ 9, 10, 11]),
            np.array([12, 13, 14]),
            np.array([15, 16, 17])
        ]

        cvindex = cvindex_groupwise(x, exclusive=a)

        for i, (tr, te) in enumerate(cvindex):
            self.assertTrue(np.array_equal(train_index[i], tr))
            self.assertTrue(np.array_equal(test_index[i], te))


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCv)
    unittest.TextTestRunner(verbosity=2).run(suite)
