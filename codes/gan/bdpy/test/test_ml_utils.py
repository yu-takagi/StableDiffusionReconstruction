'''Tests for bdpy.ml'''


from unittest import TestCase, TestLoader, TextTestRunner

import numpy as np

from bdpy import ml


class TestMl(TestCase):
    '''Tests for 'ml' module'''

    def test_make_cvindex(self):
        '''Test for make_cvindex'''
        test_input = np.array([1, 1, 2, 2, 3, 3])

        exp_output_a = np.array([[False, True,  True],
                                 [False, True,  True],
                                 [True,  False, True],
                                 [True,  False, True],
                                 [True,  True,  False],
                                 [True,  True,  False]])
        exp_output_b = np.array([[True,  False, False],
                                 [True,  False, False],
                                 [False, True,  False],
                                 [False, True,  False],
                                 [False, False, True],
                                 [False, False, True]])

        test_output_a, test_output_b = ml.make_cvindex(test_input)

        self.assertTrue((test_output_a == exp_output_a).all())
        self.assertTrue((test_output_b == exp_output_b).all())

    def test_add_bias_default(self):
        '''Test of bdpy.ml.regress.add_bias (default: axis=0)'''
        x = np.array([[100, 110, 120],
                      [200, 210, 220]])

        exp_y = np.array([[100, 110, 120],
                          [200, 210, 220],
                          [1, 1, 1]])

        test_y = ml.add_bias(x)

        np.testing.assert_array_equal(test_y, exp_y)

    def test_add_bias_axisone(self):
        '''Test of bdpy.ml.regress.add_bias (axis=1)'''
        x = np.array([[100, 110, 120],
                      [200, 210, 220]])

        exp_y = np.array([[100, 110, 120, 1],
                          [200, 210, 220, 1]])

        test_y = ml.add_bias(x, axis=1)

        np.testing.assert_array_equal(test_y, exp_y)

    def test_add_bias_axiszero(self):
        '''Test of bdpy.ml.regress.add_bias (axis=0)'''
        x = np.array([[100, 110, 120],
                      [200, 210, 220]])

        exp_y = np.array([[100, 110, 120],
                          [200, 210, 220],
                          [1, 1, 1]])

        test_y = ml.add_bias(x, axis=0)

        np.testing.assert_array_equal(test_y, exp_y)

    def test_add_bias_invalidaxis(self):
        '''Exception test of bdpy.ml.regress.add_bias
           (invalid input in 'axis')'''
        x = np.array([[100, 110, 120],
                      [200, 210, 220]])

        self.assertRaises(ValueError, lambda: ml.add_bias(x, axis=-1))

    def test_ensemble_get_majority(self):
        '''Tests of bdpy.ml.emsenble.get_majority'''
        data = np.array([[1, 3, 2, 1, 2],
                         [2, 1, 0, 0, 2],
                         [2, 1, 1, 0, 2],
                         [1, 3, 3, 1, 1],
                         [0, 2, 3, 3, 0],
                         [3, 2, 2, 2, 1],
                         [3, 1, 3, 2, 0],
                         [3, 2, 0, 3, 1]])
        # Get the major elements in each colum (axis=0) or row (axis=1).
        # The element with the smallest value will be returned when several
        # elements were the majority.
        ans_by_column = np.array([3, 1, 3, 0, 1])
        ans_by_row = np.array([1, 0, 1, 1, 0, 2, 3, 3])
        np.testing.assert_array_almost_equal(ml.get_majority(data, axis=0),
                                             ans_by_column)
        np.testing.assert_array_almost_equal(ml.get_majority(data, axis=1),
                                             ans_by_row)


if __name__ == '__main__':
    suite = TestLoader().loadTestsFromTestCase(TestMl)
    TextTestRunner(verbosity=2).run(suite)
