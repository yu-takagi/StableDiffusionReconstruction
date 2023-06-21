'''Tests for dataform'''


from unittest import TestCase, TestLoader, TextTestRunner

import numpy as np

from bdpy.dataform import load_array, save_array


class TestUtil(TestCase):
    def test_load_save_dense_array(self):
        # ndim = 1
        data = np.random.rand(10)

        save_array('./tmp/test_array_dense_ndim1.mat', data, key='testdata')
        testdata = load_array('./tmp/test_array_dense_ndim1.mat', key='testdata')

        np.testing.assert_array_equal(data, testdata)

        # ndim = 2
        data = np.random.rand(3, 2)

        save_array('./tmp/test_array_dense_ndim2.mat', data, key='testdata')
        testdata = load_array('./tmp/test_array_dense_ndim2.mat', key='testdata')

        np.testing.assert_array_equal(data, testdata)

        # ndim = 3
        data = np.random.rand(4, 3, 2)

        save_array('./tmp/test_array_dense_ndim3.mat', data, key='testdata')
        testdata = load_array('./tmp/test_array_dense_ndim3.mat', key='testdata')

        np.testing.assert_array_equal(data, testdata)

    def test_load_save_sparse_array(self):
        # ndim = 1
        data = np.random.rand(10)
        data[data < 0.8] = 0

        save_array('./tmp/test_array_sparse_ndim1.mat', data, key='testdata', sparse=True)
        testdata = load_array('./tmp/test_array_sparse_ndim1.mat', key='testdata')

        np.testing.assert_array_equal(data, testdata)

        # ndim = 2
        data = np.random.rand(3, 2)
        data[data < 0.8] = 0

        save_array('./tmp/test_array_sparse_ndim2.mat', data, key='testdata', sparse=True)
        testdata = load_array('./tmp/test_array_sparse_ndim2.mat', key='testdata')

        np.testing.assert_array_equal(data, testdata)

        # ndim = 3
        data = np.random.rand(4, 3, 2)
        data[data < 0.8] = 0

        save_array('./tmp/test_array_sparse_ndim3.mat', data, key='testdata', sparse=True)
        testdata = load_array('./tmp/test_array_sparse_ndim3.mat', key='testdata')

        np.testing.assert_array_equal(data, testdata)

    def test_load_array_jl(self):
        data = np.array([[1, 0, 0, 0],
                         [2, 2, 0, 0],
                         [3, 3, 3, 0]])

        testdata = load_array('data/array_jl_dense_v1.mat', key='a')
        np.testing.assert_array_equal(data, testdata)

        testdata = load_array('data/array_jl_sparse_v1.mat', key='a')
        np.testing.assert_array_equal(data, testdata)


if __name__ == '__main__':
    suite = TestLoader().loadTestsFromTestCase(TestUtil)
    TextTestRunner(verbosity=2).run(suite)
