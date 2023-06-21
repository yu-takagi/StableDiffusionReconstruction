'''Tests for bdpy.util'''

from unittest import TestCase, TestLoader, TextTestRunner

import os

import numpy as np
import scipy.io as sio

import bdpy.mri as bmr


class TestMri(TestCase):
    '''Tests for 'mri' module'''

    def __init__(self, *args, **kwargs):

        super(TestMri, self).__init__(*args, **kwargs)

        self.data_dir = './data/mri'
        self.test_files = ['epi0001.img', 'epi0002.img', 'epi0003.img',
                           'epi0004.img', 'epi0005.img']
        self.exp_file = 'epi.mat'

        # Get exptected data
        self.exp_dat = sio.loadmat(os.path.join(self.data_dir, self.exp_file))
        self.exp_voxdata = self.exp_dat['voxData']
        self.exp_xyz = self.exp_dat['xyz']

    def test_add_load_epi_pass0001(self):
        '''Test for load_epi (pass case 0001)'''

        # Load data
        test_voxdata, test_xyz = bmr.load_epi([os.path.join(self.data_dir, f)
                                               for f in self.test_files])

        # Matching columns b/w test and expected data
        num_voxel = self.exp_xyz.shape[1]

        exp_hash = np.dot([100, 1, 0.01], self.exp_xyz)
        index_map = -1 * np.ones(num_voxel)

        for i in range(num_voxel):
            txyz = test_xyz[:, i]
            thash = np.dot([100, 1, 0.01], txyz)

            hit_index = exp_hash == thash
            # This sometimes misses the corresponding xyz.
            # In such a case, the closest xyz will be matched.

            if hit_index.any():
                index_map[exp_hash == thash] = i
            else:
                # Matching the closest xyz
                min_diff_ind = np.argmin(np.abs(exp_hash - thash))
                index_map[min_diff_ind] = i

        test_voxdata_reord = test_voxdata[:, np.array(index_map, dtype=np.int)]
        test_xyz_reord = test_xyz[:, np.array(index_map, dtype=np.int)]

        # Compare test and exp data
        np.testing.assert_array_equal(test_voxdata_reord, self.exp_voxdata)
        np.testing.assert_array_equal(test_xyz_reord, self.exp_xyz)

    def test_get_roiflag_pass0001(self):
        '''Test for get_roiflag (pass case 0001)'''

        roi_xyz = [np.array([[1, 2, 3],
                             [1, 2, 3],
                             [1, 2, 3]])]
        epi_xyz = np.array([[1, 2, 3, 4, 5, 6],
                            [1, 2, 3, 4, 5, 6],
                            [1, 2, 3, 4, 5, 6]])

        exp_output = np.array([1, 1, 1, 0, 0, 0])

        test_output = bmr.get_roiflag(roi_xyz, epi_xyz)

        self.assertTrue((test_output == exp_output).all())

    def test_get_roiflag_pass0002(self):
        '''Test for get_roiflag (pass case 0002)'''

        roi_xyz = [np.array([[1, 2, 3],
                             [1, 2, 3],
                             [1, 2, 3]]),
                   np.array([[5, 6],
                             [5, 6],
                             [5, 6]])]
        epi_xyz = np.array([[1, 2, 3, 4, 5, 6],
                            [1, 2, 3, 4, 5, 6],
                            [1, 2, 3, 4, 5, 6]])

        exp_output = np.array([[1, 1, 1, 0, 0, 0],
                               [0, 0, 0, 0, 1, 1]])

        test_output = bmr.get_roiflag(roi_xyz, epi_xyz)

        self.assertTrue((test_output == exp_output).all())


if __name__ == '__main__':
    suite = TestLoader().loadTestsFromTestCase(TestMri)
    TextTestRunner(verbosity=2).run(suite)
