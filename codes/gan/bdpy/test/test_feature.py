from unittest import TestCase, TestLoader, TextTestRunner

import numpy as np

from bdpy.feature import normalize_feature


class TestUtilFeature(TestCase):

    def test_normalize_feature_1d(self):
        feat = np.random.rand(4096)
        feat_mean0 = np.random.rand(1, 1)
        feat_std0 = np.random.rand(1, 1)

        ddof = 1

        feat_mean_ch = np.mean(feat, axis=None, keepdims=True)
        feat_mean_all = np.mean(feat, axis=None, keepdims=True)
        feat_std_ch = np.std(feat, axis=None, ddof=ddof, keepdims=True)
        feat_std_all = np.mean(np.std(feat, axis=None, ddof=ddof, keepdims=True), keepdims=True)

        # Mean (channel-wise) + SD (channel-wise)
        feat_valid = ((feat - feat_mean_ch) / feat_std_ch) * feat_std0 + feat_mean0
        feat_test = normalize_feature(feat,
                                      channel_axis=0, channel_wise_mean=True, channel_wise_std=True,
                                      shift=feat_mean0, scale=feat_std0,
                                      std_ddof=1)

        np.testing.assert_array_equal(feat_test, feat_valid)

        # Mean (channel-wise) + SD (all)
        feat_valid = ((feat - feat_mean_ch) / feat_std_all) * feat_std0 + feat_mean0
        feat_test = normalize_feature(feat,
                                      channel_axis=0, channel_wise_mean=True, channel_wise_std=False,
                                      shift=feat_mean0, scale=feat_std0,
                                      std_ddof=1)

        np.testing.assert_array_equal(feat_test, feat_valid)

        # Mean (all) + SD (channel-wise)
        feat_valid = ((feat - feat_mean_all) / feat_std_ch) * feat_std0 + feat_mean0
        feat_test = normalize_feature(feat,
                                      channel_axis=0, channel_wise_mean=False, channel_wise_std=True,
                                      shift=feat_mean0, scale=feat_std0,
                                      std_ddof=1)

        np.testing.assert_array_equal(feat_test, feat_valid)

        # Mean (all) + SD (all)
        feat_valid = ((feat - feat_mean_all) / feat_std_all) * feat_std0 + feat_mean0
        feat_test = normalize_feature(feat,
                                      channel_axis=0, channel_wise_mean=False, channel_wise_std=False,
                                      shift=feat_mean0, scale=feat_std0,
                                      std_ddof=1)

        np.testing.assert_array_equal(feat_test, feat_valid)

        # Mean (channel-wise) + SD (channel-wise), self-mean shift
        feat_valid = ((feat - feat_mean_ch) / feat_std_ch) * feat_std0 + feat_mean_ch
        feat_test = normalize_feature(feat,
                                      channel_axis=0, channel_wise_mean=True, channel_wise_std=True,
                                      shift='self', scale=feat_std0,
                                      std_ddof=1)

        np.testing.assert_array_equal(feat_test, feat_valid)

        # Mean (channel-wise) + SD (channel-wise), self-mean shift and self-SD scale
        feat_valid = ((feat - feat_mean_ch) / feat_std_ch) * feat_std_ch + feat_mean_ch
        feat_test = normalize_feature(feat,
                                      channel_axis=0, channel_wise_mean=True, channel_wise_std=True,
                                      shift='self', scale='self',
                                      std_ddof=1)

        np.testing.assert_array_equal(feat_test, feat_valid)

    def test_normalize_feature_3d(self):
        feat = np.random.rand(64, 16, 16)
        feat_mean0 = np.random.rand(64, 1, 1)
        feat_std0 = np.random.rand(64, 1, 1)

        ddof = 1

        feat_mean_ch = np.mean(feat, axis=(1, 2), keepdims=True)
        feat_mean_all = np.mean(feat, axis=None, keepdims=True)
        feat_std_ch = np.std(feat, axis=(1, 2), ddof=ddof, keepdims=True)
        feat_std_all = np.mean(np.std(feat, axis=(1, 2), ddof=ddof, keepdims=True), keepdims=True)

        axes_along = (1, 2)

        # Mean (channel-wise) + SD (channel-wise)
        feat_valid = ((feat - feat_mean_ch) / feat_std_ch) * feat_std0 + feat_mean0
        feat_test = normalize_feature(feat,
                                      channel_axis=0, channel_wise_mean=True, channel_wise_std=True,
                                      shift=feat_mean0, scale=feat_std0,
                                      std_ddof=1)

        np.testing.assert_array_equal(feat_test, feat_valid)

        # Mean (channel-wise) + SD (all)
        feat_valid = ((feat - feat_mean_ch) / feat_std_all) * feat_std0 + feat_mean0
        feat_test = normalize_feature(feat,
                                      channel_axis=0, channel_wise_mean=True, channel_wise_std=False,
                                      shift=feat_mean0, scale=feat_std0,
                                      std_ddof=1)

        np.testing.assert_array_equal(feat_test, feat_valid)

        # Mean (all) + SD (channel-wise)
        feat_valid = ((feat - feat_mean_all) / feat_std_ch) * feat_std0 + feat_mean0
        feat_test = normalize_feature(feat,
                                      channel_axis=0, channel_wise_mean=False, channel_wise_std=True,
                                      shift=feat_mean0, scale=feat_std0,
                                      std_ddof=1)

        np.testing.assert_array_equal(feat_test, feat_valid)

        # Mean (all) + SD (all)
        feat_valid = ((feat - feat_mean_all) / feat_std_all) * feat_std0 + feat_mean0
        feat_test = normalize_feature(feat,
                                      channel_axis=0, channel_wise_mean=False, channel_wise_std=False,
                                      shift=feat_mean0, scale=feat_std0,
                                      std_ddof=1)

        np.testing.assert_array_equal(feat_test, feat_valid)

        # Mean (channel-wise) + SD (channel-wise), self-mean shift
        feat_valid = ((feat - feat_mean_ch) / feat_std_ch) * feat_std0 + feat_mean_ch
        feat_test = normalize_feature(feat,
                                      channel_axis=0, channel_wise_mean=True, channel_wise_std=True,
                                      shift='self', scale=feat_std0,
                                      std_ddof=1)

        np.testing.assert_array_equal(feat_test, feat_valid)

        # Mean (channel-wise) + SD (channel-wise), self-mean shift and self-SD scale
        feat_valid = ((feat - feat_mean_ch) / feat_std_ch) * feat_std_ch + feat_mean_ch
        feat_test = normalize_feature(feat,
                                      channel_axis=0, channel_wise_mean=True, channel_wise_std=True,
                                      shift='self', scale='self',
                                      std_ddof=1)

        # SD scaling only
        feat_valid = (feat / feat_std_all) * feat_std0
        feat_test = normalize_feature(feat,
                                      scaling_only=True,
                                      channel_wise_std=False,
                                      scale=feat_std0, std_ddof=1)

        np.testing.assert_array_equal(feat_test, feat_valid)


if __name__ == '__main__':
    suite = TestLoader().loadTestsFromTestCase(TestUtilFeature)
    TextTestRunner(verbosity=2).run(suite)
