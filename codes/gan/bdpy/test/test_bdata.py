'''Tests for bdpy.bdata'''


from unittest import TestCase, TestLoader, TextTestRunner

import copy

import numpy as np
from numpy.testing import assert_array_equal

import bdpy


class TestBdata(TestCase):
    '''Tests of 'bdata' module'''

    def __init__(self, *args, **kwargs):
        super(TestBdata, self).__init__(*args, **kwargs)

    def test_add_get(self):
        '''Test for BData.add and get.'''
        data_x = np.random.rand(5, 10)
        data_y = np.random.rand(5, 8)
        data_z = np.random.rand(5, 20)

        b = bdpy.BData()

        b.add(data_x, 'Data_X')
        b.add(data_y, 'Data_Y')
        b.add(data_z, 'Data_Z')

        # dataset
        assert_array_equal(b.get('Data_X'), data_x)
        assert_array_equal(b.get('Data_Y'), data_y)
        assert_array_equal(b.get('Data_Z'), data_z)

        # metadata
        assert_array_equal(b.metadata.get('Data_X', 'value'), np.array([1] * 10 + [np.nan] * 8 + [np.nan] * 20))
        assert_array_equal(b.metadata.get('Data_Y', 'value'), np.array([np.nan] * 10 + [1] * 8 + [np.nan] * 20))
        assert_array_equal(b.metadata.get('Data_Z', 'value'), np.array([np.nan] * 10 + [np.nan] * 8 + [1] * 20))

        # metadata (BData.get_metadata)
        assert_array_equal(b.get_metadata('Data_X'), np.array([1] * 10 + [np.nan] * 8 + [np.nan] * 20))
        assert_array_equal(b.get_metadata('Data_Y'), np.array([np.nan] * 10 + [1] * 8 + [np.nan] * 20))
        assert_array_equal(b.get_metadata('Data_Z'), np.array([np.nan] * 10 + [np.nan] * 8 + [1] * 20))

    def test_metadata_add_get(self):
        '''Test for add/get_metadata.'''

        data_x = np.random.rand(5, 10)
        data_y = np.random.rand(5, 8)

        n_col = data_x.shape[1] + data_y.shape[1]

        metadata_a = np.random.rand(n_col)
        metadata_b = np.random.rand(n_col)

        b = bdpy.BData()

        b.add(data_x, 'Data_X')
        b.add(data_y, 'Data_Y')

        b.add_metadata('Metadata_A', metadata_a)
        b.add_metadata('Metadata_B', metadata_b)

        assert_array_equal(b.metadata.get('Metadata_A', 'value'), metadata_a)
        assert_array_equal(b.metadata.get('Metadata_B', 'value'), metadata_b)
        assert_array_equal(b.get_metadata('Metadata_A'), metadata_a)
        assert_array_equal(b.get_metadata('Metadata_B'), metadata_b)

    def test_metadata_add_get_where(self):
        '''Test for add/get_metadata with where option.'''

        data_x = np.random.rand(5, 10)
        data_y = np.random.rand(5, 8)

        metadata_a = np.random.rand(10)
        metadata_b = np.random.rand(8)

        b = bdpy.BData()

        b.add(data_x, 'Data_X')
        b.add(data_y, 'Data_Y')

        b.add_metadata('Metadata_A', metadata_a, where='Data_X')
        b.add_metadata('Metadata_B', metadata_b, where='Data_Y')

        assert_array_equal(b.get_metadata('Metadata_A'), np.hstack([metadata_a, np.array([np.nan] * 8)]))
        assert_array_equal(b.get_metadata('Metadata_B'), np.hstack([np.array([np.nan] * 10), metadata_b]))
        assert_array_equal(b.get_metadata('Metadata_A', where='Data_X'), metadata_a)
        assert_array_equal(b.get_metadata('Metadata_B', where='Data_Y'), metadata_b)

    def test_set_metadatadescription_1(self):
        '''Test for set_metadatadescription.'''

        data_x = np.random.rand(5, 10)
        data_y = np.random.rand(5, 8)

        metadata_a = np.random.rand(10)
        metadata_b = np.random.rand(8)

        b = bdpy.BData()
        b.add(data_x, 'Data_X')
        b.add(data_y, 'Data_Y')
        b.add_metadata('Metadata_A', metadata_a, where='Data_X')
        b.add_metadata('Metadata_B', metadata_b, where='Data_Y')

        metadata_desc = 'Test metadata description'

        b.set_metadatadescription('Metadata_A', metadata_desc)

        self.assertEqual(b.metadata.get('Metadata_A', 'description'), metadata_desc)

    def test_select(self):
        '''Test for BData.select.'''

        data_x = np.random.rand(5, 10)
        data_y = np.random.rand(5, 5)

        b = bdpy.BData()
        b.add(data_x, 'Data_X')
        b.add(data_y, 'Data_Y')

        b.add_metadata('ROI_0:5', [1, 1, 1, 1, 1, 0, 0, 0, 0, 0], where='Data_X')
        b.add_metadata('ROI_3:8', [0, 0, 0, 1, 1, 1, 1, 1, 0, 0], where='Data_X')
        b.add_metadata('ROI_4:9', [0, 0, 0, 0, 1, 1, 1, 1, 1, 0], where='Data_X')

        assert_array_equal(b.select('Data_X'), data_x)
        assert_array_equal(b.select('Data_X = 1'), data_x)

        assert_array_equal(b.select('ROI_0:5'), data_x[:, 0:5])
        assert_array_equal(b.select('ROI_0:5 & ROI_3:8'), data_x[:, 3:5])
        assert_array_equal(b.select('ROI_0:5 = 1 & ROI_3:8 = 1'), data_x[:, 3:5])
        assert_array_equal(b.select('ROI_0:5 | ROI_3:8'), data_x[:, 0:8])
        assert_array_equal(b.select('ROI_0:5 = 1 | ROI_3:8 = 1'), data_x[:, 0:8])
        assert_array_equal(b.select('(ROI_0:5 | ROI_3:8) & ROI_4:9'), data_x[:, 4:8])
        assert_array_equal(b.select('ROI_0:5 | (ROI_3:8 & ROI_4:9)'), data_x[:, 0:8])

        assert_array_equal(b.select('ROI_*'), data_x[:, 0:9])
        assert_array_equal(b.select('ROI_0:5 + ROI_3:8'), data_x[:, 0:8])
        assert_array_equal(b.select('ROI_0:5 - ROI_3:8'), data_x[:, 0:3])

    # Tests for vmap
    def test_vmap_add_get(self):
        bdata = bdpy.BData()
        bdata.add(np.random.rand(4, 3), 'MainData')
        bdata.add(np.arange(4) + 1, 'Label')

        label_map = {1: 'label-1',
                     2: 'label-2',
                     3: 'label-3',
                     4: 'label-4'}
        label = ['label-1', 'label-2', 'label-3', 'label-4']

        bdata.add_vmap('Label', label_map)
        assert bdata.get_vmap('Label') == label_map

        # Get labels
        np.testing.assert_array_equal(bdata.get_label('Label'), label)

    def test_vmap_add_same_map(self):
        bdata = bdpy.BData()
        bdata.add(np.random.rand(4, 3), 'MainData')
        bdata.add(np.arange(4) + 1, 'Label')

        label_map = {1: 'label-1',
                     2: 'label-2',
                     3: 'label-3',
                     4: 'label-4'}
        label = ['label-1', 'label-2', 'label-3', 'label-4']

        bdata.add_vmap('Label', label_map)
        bdata.add_vmap('Label', label_map)
        assert bdata.get_vmap('Label') == label_map

        # Get labels
        np.testing.assert_array_equal(bdata.get_label('Label'), label)

    def test_vmap_errorcases(self):
        n_sample = 4

        bdata = bdpy.BData()
        bdata.add(np.random.rand(n_sample, 3), 'MainData')
        bdata.add(np.arange(n_sample) + 1, 'Label')

        label_map = {(i + 1): 'label-%04d' % (i + 1) for i in range(n_sample)}

        bdata.add_vmap('Label', label_map)

        # Vmap not found
        with self.assertRaises(ValueError):
            bdata.get_label('MainData')

        # Invalid vmap (map is not a dict)
        label_map_invalid = range(n_sample)
        with self.assertRaises(TypeError):
            bdata.add_vmap('Label', label_map_invalid)

        # Invalid vmap (key is str)
        label_map_invalid = {'label-%04d' % i: i for i in range(n_sample)}
        with self.assertRaises(TypeError):
            bdata.add_vmap('Label', label_map_invalid)

        # Inconsistent vmap
        label_map_inconsist = {i: 'label-%04d-inconsist' % i
                               for i in range(n_sample)}
        with self.assertRaises(ValueError):
            bdata.add_vmap('Label', label_map_inconsist)

    def test_vmap_add_unnecessary_vmap(self):
        bdata = bdpy.BData()
        bdata.add(np.random.rand(4, 3), 'MainData')
        bdata.add(np.arange(4) + 1, 'Label')

        label_map = {1: 'label-1',
                     2: 'label-2',
                     3: 'label-3',
                     4: 'label-4',
                     5: 'label-5'}
        label_map_ture = {1: 'label-1',
                          2: 'label-2',
                          3: 'label-3',
                          4: 'label-4'}

        bdata.add_vmap('Label', label_map)
        assert bdata.get_vmap('Label') == label_map_ture

    def test_vmap_add_insufficient_vmap(self):
        bdata = bdpy.BData()
        bdata.add(np.random.rand(4, 3), 'MainData')
        bdata.add(np.arange(4) + 1, 'Label')

        label_map = {1: 'label-1',
                     2: 'label-2',
                     3: 'label-3'}

        with self.assertRaises(ValueError):
            bdata.add_vmap('Label', label_map)

    def test_vmap_add_invalid_name_vmap(self):
        bdata = bdpy.BData()
        bdata.add(np.random.rand(4, 3), 'MainData')
        bdata.add(np.arange(4) + 1, 'Label')

        label_map = {1: 'label-1',
                     2: 'label-2',
                     3: 'label-3',
                     4: 'label-4'}

        with self.assertRaises(ValueError):
            bdata.add_vmap('InvalidLabel', label_map)


if __name__ == "__main__":
    test_suite = TestLoader().loadTestsFromTestCase(TestBdata)
    TextTestRunner(verbosity=2).run(test_suite)
