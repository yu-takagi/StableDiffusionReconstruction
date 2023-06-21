'''Tests for bdpy.stats'''


from unittest import TestCase, TestLoader, TextTestRunner

import numpy as np

import bdpy.stats as bdst


class TestStats(TestCase):
    '''Tests for bdpy.stats'''

    def test_corrcoef_matrix_matrix_default(self):
        '''Test for corrcoef (matrix and matrix, default, var=row)'''

        x = np.random.rand(100, 10)
        y = np.random.rand(100, 10)

        exp_output = np.diag(np.corrcoef(x, y)[:x.shape[0], x.shape[0]:])

        test_output = bdst.corrcoef(x, y)

        np.testing.assert_array_equal(test_output, exp_output)

    def test_corrcoef_matrix_matrix_varcol(self):
        '''Test for corrcoef (matrix and matrix, var=col)'''

        x = np.random.rand(100, 10)
        y = np.random.rand(100, 10)

        exp_output = np.diag(np.corrcoef(x, y, rowvar=0)[:x.shape[1],
                                                         x.shape[1]:])

        test_output = bdst.corrcoef(x, y, var='col')

        np.testing.assert_array_equal(test_output, exp_output)

    def test_corrcoef_vector_vector(self):
        '''Test for corrcoef (vector and vector)'''

        x = np.random.rand(100)
        y = np.random.rand(100)

        exp_output = np.corrcoef(x, y)[0, 1]

        test_output = bdst.corrcoef(x, y)

        np.testing.assert_array_equal(test_output, exp_output)

    def test_corrcoef_hvector_hvector(self):
        '''Test for corrcoef (horizontal vector and horizontal vector)'''

        x = np.random.rand(1, 100)
        y = np.random.rand(1, 100)

        exp_output = np.corrcoef(x, y)[0, 1]

        test_output = bdst.corrcoef(x, y)

        np.testing.assert_array_equal(test_output, exp_output)

    def test_corrcoef_vvector_vvector(self):
        '''Test for corrcoef (vertical vector and vertical vector)'''

        x = np.random.rand(100, 1)
        y = np.random.rand(100, 1)

        exp_output = np.corrcoef(x.T, y.T)[0, 1]

        test_output = bdst.corrcoef(x, y)

        np.testing.assert_array_equal(test_output, exp_output)

    def test_corrcoef_matrix_vector_varrow(self):
        '''Test for corrcoef (matrix and vector, var=row)'''

        x = np.random.rand(100, 10)
        y = np.random.rand(10)

        exp_output = np.corrcoef(y, x)[0, 1:]

        test_output = bdst.corrcoef(x, y)

        np.testing.assert_array_almost_equal(test_output, exp_output)

    def test_corrcoef_matrix_vector_varcol(self):
        '''Test for corrcoef (matrix and vector, var=col)'''

        x = np.random.rand(100, 10)
        y = np.random.rand(100)

        exp_output = np.corrcoef(y, x, rowvar=0)[0, 1:]

        test_output = bdst.corrcoef(x, y, var='col')

        np.testing.assert_array_almost_equal(test_output, exp_output)

    def test_corrcoef_vector_matrix_varrow(self):
        '''Test for corrcoef (vector and matrix, var=row)'''

        x = np.random.rand(10)
        y = np.random.rand(100, 10)

        exp_output = np.corrcoef(x, y)[0, 1:]

        test_output = bdst.corrcoef(x, y)

        np.testing.assert_array_almost_equal(test_output, exp_output)

    def test_corrcoef_vector_matrix_varcol(self):
        '''Test for corrcoef (vector and matrix, var=col)'''

        x = np.random.rand(100)
        y = np.random.rand(100, 10)

        exp_output = np.corrcoef(x, y, rowvar=0)[0, 1:]

        test_output = bdst.corrcoef(x, y, var='col')

        np.testing.assert_array_almost_equal(test_output, exp_output)

    def test_corrmat_default(self):
        '''Test for corrmat (default, var=row)'''

        x = np.random.rand(100, 10)
        y = np.random.rand(100, 10)

        exp_output = np.corrcoef(x, y)[:x.shape[0], x.shape[0]:]

        test_output = bdst.corrmat(x, y)

        np.testing.assert_array_almost_equal(test_output, exp_output)

    def test_corrmat_varcol(self):
        '''Test for corrmat (var=col)'''

        x = np.random.rand(100, 10)
        y = np.random.rand(100, 10)

        exp_output = np.corrcoef(x, y, rowvar=0)[:x.shape[1], x.shape[1]:]

        test_output = bdst.corrmat(x, y, var='col')

        np.testing.assert_array_almost_equal(test_output, exp_output)


if __name__ == '__main__':
    suite = TestLoader().loadTestsFromTestCase(TestStats)
    TextTestRunner(verbosity=2).run(suite)
