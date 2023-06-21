'''Caffe module.'''


import os

import PIL
import caffe
import numpy as np
from bdpy.dataform import save_array
from tqdm import tqdm


def extract_image_features(image_file, net, layers=[], crop_center=False, image_preproc=[], save_dir=None, verbose=False, progbar=False, return_features=True):
    '''
    Extract DNN features of a given image.

    Parameters
    ----------
    image_file : str or list
      (List of) path to the input image file(s).
    net : Caffe network instance
    layers : list
      List of DNN layers of which features are returned.
    crop_center : bool (default: False)
      Crop the center of an image or not.
    image_preproc : list (default: [])
      List of additional preprocessing functions. The function input/output
      should be a PIL.Image instance. The preprocessing functions are applied
      after RGB conversion, center-cropping, and resizing of the input image.
    save_dir : None or str (default: None)
      Save the features in the specified directory if not None.
    verbose : bool (default: False)
      Output verbose messages or not.
    return_features: bool (default: True)
      Return the extracted features or not.

    Returns
    -------
    dict
      Dictionary in which keys are DNN layers and values are features.
    '''

    if isinstance(image_file, str):
        image_file = [image_file]

    features_dict = {}

    if progbar:
        image_file = tqdm(image_file)

    for imgf in image_file:
        if verbose:
            print('Image:  %s' % imgf)

        image_size = net.blobs['data'].data.shape[-2:]
        mean_img = net.transformer.mean['data']

        # Open the image
        img = PIL.Image.open(imgf)

        # Convert non-RGB to RGB
        if img.mode == 'CMYK':
            img = img.convert('RGB')

        if img.mode == 'RGBA':
            bg = PIL.Image.new('RGB', img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg

        # Convert monochrome to RGB
        if img.mode == 'L':
            img = img.convert('RGB')

        # Center cropping
        if crop_center:
            w, h = img.size
            img = img.crop(((w - min(img.size)) // 2,
                            (h - min(img.size)) // 2,
                            (w + min(img.size)) // 2,
                            (h + min(img.size)) // 2))

        # Resize
        img = img.resize(image_size, PIL.Image.BICUBIC)

        for p in image_preproc:
            img = p(img)

        img_array = np.array(img)

        try:
            img_array = np.float32(np.transpose(img_array, (2, 0, 1))[::-1]) - np.reshape(mean_img, (3, 1, 1))
        except:
            import pdb; pdb.set_trace()

        # Forwarding
        net.blobs['data'].reshape(1, 3, img_array.shape[1], img_array.shape[2])
        net.blobs['data'].data[0] = img_array
        net.forward()

        # Get features
        for lay in layers:
            feat = net.blobs[lay].data.copy()

            if return_features:
                if lay in features_dict:
                    features_dict.update({
                        lay: np.vstack([features_dict[lay], feat])
                    })
                else:
                    features_dict.update({lay: feat})

            if not save_dir is None:
                # Save the features
                save_dir_lay = os.path.join(save_dir, lay.replace('/', ':'))
                save_file = os.path.join(save_dir_lay,
                                         os.path.splitext(os.path.basename(imgf))[0] + '.mat')
                if not os.path.exists(save_dir_lay):
                    os.makedirs(save_dir_lay)
                if os.path.exists(save_file):
                    if verbose:
                        print('%s already exists. Skipped.' % save_file)
                    continue
                save_array(save_file, feat, key='feat', dtype=np.float32, sparse=False)
                if verbose:
                    print('%s saved.' % save_file)

    if return_features:
        return features_dict
    else:
        return None
