'''Reconstruction utilities.'''


import numpy as np
import scipy.ndimage as nd


def clip_extreme(x, pct=1):
    '''
    Clip extreme values.

    Original version was written by Shen Guo-Hua.
    '''

    if pct < 0:
        pct = 0.

    if pct > 100:
        pct = 100.

    x = np.clip(x,
                np.percentile(x, pct / 2.),
                np.percentile(x, 100 - pct / 2.))
    return x


def gaussian_blur(img, sigma):
    '''
    Smooth the image with gaussian filter.

    Original version was written by Shen Guo-Hua.
    '''

    if sigma > 0:
        img[0] = nd.filters.gaussian_filter(img[0], sigma, order=0)
        img[1] = nd.filters.gaussian_filter(img[1], sigma, order=0)
        img[2] = nd.filters.gaussian_filter(img[2], sigma, order=0)
    return img


def image_norm(img):
    '''
    Calculate the norm of the RGB for each pixel.

    Original version was written by Shen Guo-Hua.
    '''
    img_norm = np.sqrt(img[0] ** 2 + img[1] ** 2 + img[2] ** 2)
    return img_norm


def normalize_image(img):
    '''
    Normalize the image.

    Map the minimum pixel to 0; map the maximum pixel to 255.
    Convert the pixels to be int.


    Original version was written by Shen Guo-Hua.
    '''

    img = img - img.min()
    if img.max() > 0:
        img = img * (255.0 / img.max())
    img = np.uint8(img)
    return img


def make_feature_masks(features, masks, channels):
    '''Make feature masks.


    Parameters
    ----------
    features : dict
        A python dictionary consists of CNN features of target layers,
        arranged in pairs of layer name (key) and CNN features
        (value).
    masks : dict, optional
        A python dictionary consists of masks for CNN features,
        arranged in pairs of layer name (key) and mask (value); the
        mask selects units for each layer to be used in the loss
        function (1: using the uint; 0: excluding the unit); mask can
        be 3D or 2D numpy array; use all the units if some layer not
        in the dictionary; setting to None for using all units for all
        layers.
    channels : dict, optional
        A python dictionary consists of channels to be selected,
        arranged in pairs of layer name (key) and channel numbers
        (value); the channel numbers of each layer are the channels to
        be used in the loss function; use all the channels if the some
        layer not in the dictionary; setting to None for using all
        channels for all layers.

    Returns
    -------
    feature_masks : dict
        A python dictionary consists of masks for CNN features,
        arranged in pairs of layer name (key) and mask (value); mask
        has the same shape as the CNN features of the corresponding
        layer;

    Note
    ----
    Original version was written by Shen Guo-Hua.

    '''

    feature_masks = {}
    for layer in features.keys():
        if (masks is None or masks == {} or masks == [] or (layer not in masks.keys())) and (channels is None or channels == {} or channels == [] or (layer not in channels.keys())):  # use all features and all channels
            feature_masks[layer] = np.ones_like(features[layer])
        elif isinstance(masks, dict) and (layer in masks.keys()) and isinstance(masks[layer], np.ndarray) and masks[layer].ndim == 3 and masks[layer].shape[0] == features[layer].shape[0] and masks[layer].shape[1] == features[layer].shape[1] and masks[layer].shape[2] == features[layer].shape[2]:  # 3D mask
            feature_masks[layer] = masks[layer]
        # 1D feat and 1D mask
        elif isinstance(masks, dict) and (layer in masks.keys()) and isinstance(masks[layer], np.ndarray) and features[layer].ndim == 1 and masks[layer].ndim == 1 and masks[layer].shape[0] == features[layer].shape[0]:
            feature_masks[layer] = masks[layer]
        elif (masks is None or masks == {} or masks == [] or (layer not in masks.keys())) and isinstance(channels, dict) and (layer in channels.keys()) and isinstance(channels[layer], np.ndarray) and channels[layer].size > 0:  # select channels
            mask_2D = np.ones_like(features[layer][0])
            mask_3D = np.tile(mask_2D, [len(channels[layer]), 1, 1])
            feature_masks[layer] = np.zeros_like(features[layer])
            feature_masks[layer][channels[layer], :, :] = mask_3D
        # use 2D mask select features for all channels
        elif isinstance(masks, dict) and (layer in masks.keys()) and isinstance(masks[layer], np.ndarray) and masks[layer].ndim == 2 and (channels is None or channels == {} or channels == [] or (layer not in channels.keys())):
            mask_2D_0 = masks[layer]
            mask_size0 = mask_2D_0.shape
            mask_size = features[layer].shape[1:]
            if mask_size0[0] == mask_size[0] and mask_size0[1] == mask_size[1]:
                mask_2D = mask_2D_0
            else:
                mask_2D = np.ones(mask_size)
                n_dim1 = min(mask_size0[0], mask_size[0])
                n_dim2 = min(mask_size0[1], mask_size[1])
                idx0_dim1 = np.arange(n_dim1) + \
                    round((mask_size0[0] - n_dim1)/2)
                idx0_dim2 = np.arange(n_dim2) + \
                    round((mask_size0[1] - n_dim2)/2)
                idx_dim1 = np.arange(n_dim1) + round((mask_size[0] - n_dim1)/2)
                idx_dim2 = np.arange(n_dim2) + round((mask_size[1] - n_dim2)/2)
                mask_2D[idx_dim1, idx_dim2] = mask_2D_0[idx0_dim1, idx0_dim2]
            feature_masks[layer] = np.tile(
                mask_2D, [features[layer].shape[0], 1, 1])
        else:
            feature_masks[layer] = 0

    return feature_masks
