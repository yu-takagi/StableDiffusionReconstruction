'''Test for bdpy.vstack'''


from unittest import TestCase, TestLoader, TextTestRunner

import numpy as np

import bdpy
from bdpy import vstack, metadata_equal


class TestVstack(TestCase):

    def test_vstack(self):
        x0_data = np.random.rand(10, 20)
        x0_label = np.random.rand(10, 1)

        x1_data = np.random.rand(10, 20)
        x1_label = np.random.rand(10, 1)

        bdata0 = bdpy.BData()
        bdata0.add(x0_data,  'Data')
        bdata0.add(x0_label, 'Label')

        bdata1 = bdpy.BData()
        bdata1.add(x1_data, 'Data')
        bdata1.add(x1_label, 'Label')

        bdata_merged = vstack([bdata0, bdata1])

        np.testing.assert_array_equal(bdata_merged.select('Data'),
                                      np.vstack([x0_data, x1_data]))
        np.testing.assert_array_equal(bdata_merged.select('Label'),
                                      np.vstack([x0_label, x1_label]))

    def test_vstack_successive(self):
        x0_data = np.random.rand(10, 20)
        x0_label = np.random.rand(10, 1)
        x0_run = np.arange(10).reshape(10, 1) + 1

        x1_data = np.random.rand(10, 20)
        x1_label = np.random.rand(10, 1)
        x1_run = np.arange(10).reshape(10, 1) + 1

        bdata0 = bdpy.BData()
        bdata0.add(x0_data,  'Data')
        bdata0.add(x0_label, 'Label')
        bdata0.add(x0_run, 'Run')

        bdata1 = bdpy.BData()
        bdata1.add(x1_data, 'Data')
        bdata1.add(x1_label, 'Label')
        bdata1.add(x0_run, 'Run')

        bdata_merged = vstack([bdata0, bdata1], successive=['Run'])

        np.testing.assert_array_equal(bdata_merged.select('Data'),
                                      np.vstack([x0_data, x1_data]))
        np.testing.assert_array_equal(bdata_merged.select('Label'),
                                      np.vstack([x0_label, x1_label]))
        np.testing.assert_array_equal(bdata_merged.select('Run'),
                                      np.vstack([x0_run,
                                                 x1_run + len(x0_run)]))

    def test_vstack_minimal(self):
        x0_data = np.random.rand(5, 10)
        x0_label = np.random.rand(5, 1)

        x1_data = np.random.rand(5, 10)
        x1_label = np.random.rand(5, 1)

        bdata0 = bdpy.BData()
        bdata0.add(x0_data,  'Data')
        bdata0.add(x0_label, 'Label')
        bdata0.add_metadata('key shared', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.nan], 'Shared meta-data')
        bdata0.add_metadata('key only in 0', np.random.rand(11), 'Meta-data only in bdata0')


        bdata1 = bdpy.BData()
        bdata1.add(x1_data, 'Data')
        bdata1.add(x1_label, 'Label')
        bdata1.add_metadata('key shared', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.nan], 'Shared meta-data')
        bdata0.add_metadata('key only in 1', np.random.rand(11), 'Meta-data only in bdata1')

        bdata_merged = vstack([bdata0, bdata1], metadata_merge='minimal')

        np.testing.assert_array_equal(bdata_merged.select('Data'),
                                      np.vstack([x0_data, x1_data]))
        np.testing.assert_array_equal(bdata_merged.select('Label'),
                                      np.vstack([x0_label, x1_label]))
        np.testing.assert_array_equal(bdata_merged.get_metadata('key shared'),
                                      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.nan])
        self.assertFalse('key only in 0' in bdata_merged.metadata.key)
        self.assertFalse('key only in 1' in bdata_merged.metadata.key)

    def test_vstack_vmap(self):
        x0_data = np.random.rand(10, 20)
        x0_label = np.random.permutation(np.arange(10)).reshape(10, 1) + 1

        x0_label_map = {k: 'label_%04d' % k for k in x0_label.flatten()}

        x1_data = np.random.rand(10, 20)
        x1_label = np.random.permutation(np.arange(10)).reshape(10, 1) + 1

        x1_label_map = {k: 'label_%04d' % k for k in x1_label.flatten()}

        bdata0 = bdpy.BData()
        bdata0.add(x0_data,  'Data')
        bdata0.add(x0_label, 'Label')
        bdata0.add_vmap('Label', x0_label_map)

        bdata1 = bdpy.BData()
        bdata1.add(x1_data, 'Data')
        bdata1.add(x1_label, 'Label')
        bdata1.add_vmap('Label', x1_label_map)

        bdata_merged = vstack([bdata0, bdata1])

        np.testing.assert_array_equal(bdata_merged.select('Data'),
                                      np.vstack([x0_data, x1_data]))
        np.testing.assert_array_equal(bdata_merged.select('Label'),
                                      np.vstack([x0_label, x1_label]))

        # Check vmap
        assert bdata0.get_vmap('Label') == bdata1.get_vmap('Label')
        assert bdata_merged.get_vmap('Label') == bdata0.get_vmap('Label')

    def test_vstack_vmap_merge_diff_vmap(self):
        x0_data = np.random.rand(10, 20)
        x0_label = np.random.permutation(np.arange(10)).reshape(10, 1) + 1

        x0_label_map = {k: 'label_%04d' % k for k in x0_label.flatten()}

        x1_data = np.random.rand(10, 20)
        x1_label = np.random.permutation(np.arange(10)).reshape(10, 1) + 11

        x1_label_map = {k: 'label_%04d' % k for k in x1_label.flatten()}

        bdata0 = bdpy.BData()
        bdata0.add(x0_data,  'Data')
        bdata0.add(x0_label, 'Label')
        bdata0.add_vmap('Label', x0_label_map)

        bdata1 = bdpy.BData()
        bdata1.add(x1_data, 'Data')
        bdata1.add(x1_label, 'Label')
        bdata1.add_vmap('Label', x1_label_map)

        bdata_merged = vstack([bdata0, bdata1])

        np.testing.assert_array_equal(bdata_merged.select('Data'),
                                      np.vstack([x0_data, x1_data]))
        np.testing.assert_array_equal(bdata_merged.select('Label'),
                                      np.vstack([x0_label, x1_label]))

        # Check vmap
        merged_vmap = {}
        merged_vmap.update(x0_label_map)
        merged_vmap.update(x1_label_map)
        assert bdata_merged.get_vmap('Label') == merged_vmap

    def test_vstack_vmap_inconsistent_vmap(self):
        x0_data = np.random.rand(10, 20)
        x0_label = np.random.permutation(np.arange(10)).reshape(10, 1) + 1

        x0_label_map = {k: 'label_%04d' % k for k in x0_label.flatten()}

        x1_data = np.random.rand(10, 20)
        x1_label = np.random.permutation(np.arange(10)).reshape(10, 1) + 1

        x1_label_map = {k: 'label_%04d_diff' % k for k in x1_label.flatten()}

        bdata0 = bdpy.BData()
        bdata0.add(x0_data,  'Data')
        bdata0.add(x0_label, 'Label')
        bdata0.add_vmap('Label', x0_label_map)

        bdata1 = bdpy.BData()
        bdata1.add(x1_data, 'Data')
        bdata1.add(x1_label, 'Label')
        bdata1.add_vmap('Label', x1_label_map)

        with self.assertRaises(ValueError):
            vstack([bdata0, bdata1], successive=['Label'])

    def test_metadata_equal(self):
        bdata0 = bdpy.BData()
        bdata1 = bdpy.BData()

        bdata0.add_metadata('key 0', [1, 1, 0], 'Test key 0')
        bdata1.add_metadata('key 0', [1, 1, 0], 'Test key 0')

        bdata0.add_metadata('key 1', [0, 0, 1], 'Test key 1')
        bdata1.add_metadata('key 1', [0, 0, 1], 'Test key 1')

        self.assertTrue(metadata_equal(bdata0, bdata1))
        self.assertTrue(metadata_equal(bdata0, bdata1, strict=False))
        self.assertTrue(metadata_equal(bdata0, bdata1, strict=True))

    def test_metadata_equal_notequal_key(self):
        bdata0 = bdpy.BData()
        bdata1 = bdpy.BData()

        bdata0.add_metadata('key 0', [1, 1, 0], 'Test key 0')
        bdata1.add_metadata('key 0', [1, 1, 0], 'Test key 0')

        bdata0.add_metadata('key 1', [0, 0, 1], 'Test key 1')
        bdata1.add_metadata('key 2', [0, 0, 1], 'Test key 1')

        self.assertFalse(metadata_equal(bdata0, bdata1))
        self.assertFalse(metadata_equal(bdata0, bdata1, strict=False))
        self.assertFalse(metadata_equal(bdata0, bdata1, strict=False))

    def test_metadata_equal_notequal_value(self):
        bdata0 = bdpy.BData()
        bdata1 = bdpy.BData()

        bdata0.add_metadata('key 0', [1, 1, 0], 'Test key 0')
        bdata1.add_metadata('key 0', [1, 1, 0], 'Test key 0')

        bdata0.add_metadata('key 1', [0, 0, 1], 'Test key 1')
        bdata1.add_metadata('key 1', [0, 0, 2], 'Test key 1')

        self.assertFalse(metadata_equal(bdata0, bdata1))
        self.assertFalse(metadata_equal(bdata0, bdata1, strict=False))
        self.assertFalse(metadata_equal(bdata0, bdata1, strict=False))

    def test_metadata_equal_loose(self):
        bdata0 = bdpy.BData()
        bdata1 = bdpy.BData()

        bdata0.add_metadata('key 0', [1, 1, 0], 'Test key 0')
        bdata0.add_metadata('key 1', [0, 0, 1], 'Test key 1')

        bdata1.add_metadata('key 1', [0, 0, 1], 'Test key 1')
        bdata1.add_metadata('key 0', [1, 1, 0], 'Test key 0')

        self.assertTrue(metadata_equal(bdata0, bdata1))
        self.assertTrue(metadata_equal(bdata0, bdata1, strict=False))
        self.assertFalse(metadata_equal(bdata0, bdata1, strict=True))

    def test_metadata_equal_loose_notequal_key(self):
        bdata0 = bdpy.BData()
        bdata1 = bdpy.BData()

        bdata0.add_metadata('key 0', [1, 1, 0], 'Test key 0')
        bdata0.add_metadata('key 1', [0, 0, 1], 'Test key 1')

        bdata1.add_metadata('key 2', [0, 0, 1], 'Test key 1')
        bdata1.add_metadata('key 0', [1, 1, 0], 'Test key 0')

        self.assertFalse(metadata_equal(bdata0, bdata1))
        self.assertFalse(metadata_equal(bdata0, bdata1, strict=False))
        self.assertFalse(metadata_equal(bdata0, bdata1, strict=False))

    def test_metadata_equal_loose_notequal_value(self):
        bdata0 = bdpy.BData()
        bdata1 = bdpy.BData()

        bdata0.add_metadata('key 0', [1, 1, 0], 'Test key 0')
        bdata0.add_metadata('key 1', [0, 0, 1], 'Test key 1')

        bdata1.add_metadata('key 1', [0, 0, 2], 'Test key 1')
        bdata1.add_metadata('key 0', [1, 1, 0], 'Test key 0')

        self.assertFalse(metadata_equal(bdata0, bdata1))
        self.assertFalse(metadata_equal(bdata0, bdata1, strict=False))
        self.assertFalse(metadata_equal(bdata0, bdata1, strict=False))


if __name__ == "__main__":
    test_suite = TestLoader().loadTestsFromTestCase(TestVstack)
    TextTestRunner(verbosity=2).run(test_suite)
