import numpy as np


def normalize_feature(feature,
                      channel_wise_mean=True, channel_wise_std=True,
                      channel_axis=0,
                      std_ddof=1,
                      shift=None, scale=None,
                      scaling_only=False):
    '''Normalize feature.

    Parameters
    ----------
    feature : ndarray
        Feature to be normalized.
    channel_wise_mean, channel_wise_std : bool (default: True)
        If `True`, run channel-wise mean/SD normalization.
    channel_axis : int (default: 0)
        Channel axis.
    shift, scale : None, 'self', or ndarray (default: None)
        If shift/scale is `None`, nothing will be added/multiplied to the normalized features.
        If `'self'`, mean/SD of `feature` will be added/multiplied to the normalized features.
        If ndarrays are given, the arrays will be added/multiplied to the normalized features.
    std_ddof : int (default: 1)
        Delta degree of freedom for SD.

    Returns
    -------
    ndarray
        Normalized (and scaled/shifted) features.
    '''

    if feature.ndim == 1:
        axes_along = None
    else:
        axes = list(range(feature.ndim))
        axes.remove(channel_axis)
        axes_along = tuple(axes)

    if channel_wise_mean:
        feat_mean = np.mean(feature, axis=axes_along, keepdims=True)
    else:
        feat_mean = np.mean(feature, keepdims=True)

    if channel_wise_std:
        feat_std = np.std(feature, axis=axes_along, ddof=std_ddof, keepdims=True)
    else:
        feat_std = np.mean(np.std(feature, axis=axes_along, ddof=std_ddof, keepdims=True), keepdims=True)

    if isinstance(shift, str) and shift == 'self':
        shift = feat_mean

    if isinstance(scale, str) and scale == 'self':
        scale = feat_std

    if scaling_only:
        feat_n = (feature / feat_std) * scale
    else:
        feat_n = ((feature - feat_mean) / feat_std)

        if not scale is None:
            feat_n = feat_n * scale
        if not shift is None:
            feat_n = feat_n + shift

    if not feature.shape == feat_n.shape:
        try:
            feat_n.reshape(feature.shape)
        except:
            raise ValueError('Invalid shape of normalized features (original: %s, normalized: %s). '
                             + 'Possibly incorrect shift and/or scale.'
                             % (str(feature.shape), str(feat_n.shape)))

    return feat_n
