'''iCNN reconstruction; gradient descent, with image generator'''
import glob
from itertools import product
import os

from bdpy.recon.torch.icnn import reconstruct
from bdpy.recon.utils import normalize_image, clip_extreme
from bdpy.dl.torch.models import VGG19, AlexNetGenerator, layer_map
from bdpy.dataform import Features, DecodedFeatures
from bdpy.feature import normalize_feature
from bdpy.util import dump_info
import numpy as np
import PIL.Image
import scipy.io as sio
import torch
import torch.optim as optim


# Main function ##############################################################

def recon_icnn_image_vgg19_dgn_relu7gen_dg(
        layers,
        features_dir,
        output_dir='./recon',
        subjects=None, rois=None,
        n_iter=200,
        device='cuda:0'
):
    '''
    - iCNN reconstruction with DGN and gradient descent
    - Encoder: VGG-19
    - Generator: AlexNetGenerator, ReLU7
    - Set `subjects` and `rois` as `[None]` for true features.
    '''



    # Network settings -------------------------------------------------------

    encoder_param_file = './models/pytorch/VGG_ILSVRC_19_layers/VGG_ILSVRC_19_layers.pt'

    '''
    layers = [
        'conv1_1', 'conv1_2', 'conv2_1', 'conv2_2',
        'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
        'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4',
        'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4',
        'fc6', 'fc7', 'fc8'
    ]
    '''
    layer_mapping = layer_map('vgg19')

    encoder_input_shape = (224, 224, 3)

    generator_param_file = './models/pytorch/bvlc_reference_caffenet_generator_ILSVRC2012_Training/generator_relu7.pt'

    # Average image of ImageNet
    image_mean_file = './models/pytorch/VGG_ILSVRC_19_layers/ilsvrc_2012_mean.npy'
    image_mean = np.load(image_mean_file)
    image_mean = np.float32([image_mean[0].mean(), image_mean[1].mean(), image_mean[2].mean()])

    feature_std_file = './models/pytorch/VGG_ILSVRC_19_layers/estimated_cnn_feat_std_VGG_ILSVRC_19_layers_ImgSize_224x224_chwise_dof1.mat'

    feature_range_file = './models/pytorch/bvlc_reference_caffenet_generator_ILSVRC2012_Training/act_range/3x/fc7.txt'

    # Delta degrees of freedom when calculating SD
    # This should be match to the DDoF used in calculating
    # SD of true DNN features (`feat_std0`)
    std_ddof = 1

    # Axis for channel in the DNN feature array
    channel_axis = 0

    # Reconstruction options -------------------------------------------------

    opts = {
        # Loss function
        'loss_func': torch.nn.MSELoss(reduction='sum'),

        # The total number of iterations for gradient descend
        'n_iter': n_iter,

        # Learning rate
        'lr': (2., 1e-10),

        # Gradient with momentum
        'momentum': (0.9, 0.9),

        # Pixel decay for each iteration
        'decay': (0.01, 0.01),

        # Use image smoothing or not
        'blurring': False,

        # A python dictionary consists of channels to be selected, arranged in
        # pairs of layer name (key) and channel numbers (value); the channel
        # numbers of each layer are the channels to be used in the loss function;
        # use all the channels if some layer not in the dictionary; setting to None
        # for using all channels for all layers;
        'channels': None,

        # A python dictionary consists of masks for the traget CNN features,
        # arranged in pairs of layer name (key) and mask (value); the mask selects
        # units for each layer to be used in the loss function (1: using the uint;
        # 0: excluding the unit); mask can be 3D or 2D numpy array; use all the
        # units if some layer not in the dictionary; setting to None for using all
        # units for all layers;
        'masks': None,

        # Display the information on the terminal for every n iterations
        'disp_interval': 1,
    }


    # Main #######################################################################

    # Initialize DNN ---------------------------------------------------------

    # Initial image for the optimization (here we use the mean of ilsvrc_2012_mean.npy as RGB values)
    initial_image = np.zeros((224, 224, 3), dtype='float32')
    initial_image[:, :, 0] = image_mean[2].copy()
    initial_image[:, :, 1] = image_mean[1].copy()
    initial_image[:, :, 2] = image_mean[0].copy()

    # Feature SD estimated from true DNN features of 10000 images
    feat_std0 = sio.loadmat(feature_std_file)

    # Feature upper/lower bounds
    cols = 4096
    up_size = (4096,)
    upper_bound = np.loadtxt(feature_range_file,
                             delimiter=' ',
                             usecols=np.arange(0, cols),
                             unpack=True)
    upper_bound = upper_bound.reshape(up_size)

    # Initial features -------------------------------------------------------
    initial_gen_feat = np.random.normal(0, 1, (4096,))

    # Setup results directory ------------------------------------------------
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save runtime information -----------------------------------------------
    dump_info(output_dir, script=__file__)

    # Set reconstruction options ---------------------------------------------
    opts.update({
        # The initial image for the optimization (setting to None will use random noise as initial image)
        'initial_feature': initial_gen_feat,
        'feature_upper_bound': upper_bound,
        'feature_lower_bound': 0.,
    })

    # Save the optional parameters
    # with open(os.path.join(output_dir, 'options.pkl'), 'w') as f:
    #     pickle.dump(opts, f)

    # Reconstrucion ----------------------------------------------------------
    for subject, roi in product(subjects, rois):

        decoded = subject is not None and roi is not None

        print('----------------------------------------')
        if decoded:
            print('Subject: ' + subject)
            print('ROI:     ' + roi)
        print('')

        if decoded:
            save_dir = os.path.join(output_dir, subject, roi)
        else:
            save_dir = os.path.join(output_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Get images if images is None
        print(os.path.join(features_dir, layers[0], subject, roi, '*.mat'))
        if decoded:
            matfiles = glob.glob(os.path.join(features_dir, layers[0], subject, roi, '*.mat'))
        else:
            matfiles = glob.glob(os.path.join(features_dir, layers[0], '*.mat'))
        t_mat_imagenames = [os.path.splitext(os.path.basename(fl))[0] for fl in matfiles]
        images = []
        for iii_img in range(len(t_mat_imagenames)):
            t_imgfile = t_mat_imagenames[iii_img]
            t_imgfile = t_imgfile[-6::]
            images.append(t_imgfile)

        # Load DNN features
        if decoded:
            features = DecodedFeatures(os.path.join(features_dir), squeeze=False)
        else:
            features = Features(features_dir)

        # Images loop
        for t_image_label in images:

            print('Image: ' + t_image_label)

            # Encoder model
            encoder = VGG19()
            encoder.to(device)
            encoder.load_state_dict(torch.load(encoder_param_file))
            encoder.eval()

            # Generator model
            generator = AlexNetGenerator()
            generator.to(device)
            generator.load_state_dict(torch.load(generator_param_file))
            generator.eval()

            # Districuted computation control
            snapshots_dir = os.path.join(save_dir, 'snapshots', 'image-%s' % t_image_label)
            if os.path.exists(snapshots_dir):
                print('Already done or running. Skipped.')
                continue

            # Load DNN features
            if decoded:
                feat = dict()
                for layer in layers:
                    image_label = 'VGG19-{:s}-{:s}-{:s}-{:s}'.format(layer, subject, roi, t_image_label)
                    t_feat = features.get(layer=layer, subject=subject, roi=roi, image=image_label)
                    feat[layer] = np.squeeze(t_feat)
                '''
                feat = {
                    layer: features.get(layer=layer, subject=subject, roi=roi, image=image_label)
                    for layer in layers
                }
                '''
            else:
                labels = features.labels
                feat = {
                    layer: features.get_features(layer)[np.array(labels) == image_label]
                    for layer in layers
                }

            #----------------------------------------
            # Normalization of decoded features
            #----------------------------------------
            normfeat = 1
            if normfeat == 1:
                for layer, ft in feat.items():

                    # Here, the decoded features are normalized and shifted/scaled
                    # as below:
                    #
                    # ft' = {(ft - mu) / mean(sd)} * mean(sd_base) + mu
                    #
                    # where
                    #
                    #   - ft       raw decoded features
                    #   - ft'      normalized and scaled decoded features
                    #   - mu       mean of decoded features across units in a
                    #              given layer
                    #   - sd       channel-wise SD of decoded features
                    #   - sd_base  channel-wise SD of features of ImageNet
                    #              Base10000 images

                    ft = [ft]

                    ft0 = normalize_feature(
                        ft[0],
                        channel_wise_mean=False, channel_wise_std=False,
                        channel_axis=channel_axis,
                        shift='self', scale=np.mean(feat_std0[layer]),
                        std_ddof=std_ddof
                    )
                    ft = ft0[np.newaxis]
                    feat.update({layer: ft})

            # Norm of the DNN features for each layer
            feat_norm = np.array([np.linalg.norm(feat[layer])
                                  for layer in layers],
                                 dtype='float32')

            # Weight of each layer in the total loss function
            # Use the inverse of the squared norm of the DNN features as the
            # weight for each layer
            weights = 1. / (feat_norm ** 2)

            # Normalise the weights such that the sum of the weights = 1
            weights = weights / weights.sum()
            layer_weights = dict(zip(layers, weights))

            opts.update({'layer_weights': layer_weights})

            # Reconstruction
            snapshots_dir = os.path.join(save_dir, 'snapshots', 'image-%s' % image_label)
            recon_image, loss_list = reconstruct(feat,
                                                 encoder,
                                                 generator=generator,
                                                 layer_mapping=layer_mapping,
                                                 optimizer=optim.SGD,
                                                 image_size=encoder_input_shape,
                                                 crop_generator_output=True,
                                                 preproc=image_preprocess,
                                                 postproc=image_deprocess,
                                                 output_dir=save_dir,
                                                 save_snapshot=False,
                                                 snapshot_dir=snapshots_dir,
                                                 snapshot_ext='tiff',
                                                 snapshot_postprocess=normalize_image,
                                                 return_loss=True,
                                                 device=device,
                                                 **opts)

            # Save the results

            # Save the raw reconstructed image
            recon_image_mat_file = os.path.join(save_dir, 'recon_image' + '-' + image_label + '.mat')
            recon_image_norm = normalize_image(clip_extreme(recon_image, pct=4))
            sio.savemat(recon_image_mat_file, {'recon_image': recon_image,'recon_image_norm': recon_image_norm})

            # To better display the image, clip pixels with extreme values (0.02% of
            # pixels with extreme low values and 0.02% of the pixels with extreme high
            # values). And then normalise the image by mapping the pixel value to be
            # within [0,255].
            recon_image_normalized_file = os.path.join(save_dir, 'recon_image_normalized' + '-' + image_label + '.tiff')
            PIL.Image.fromarray(recon_image_norm).save(recon_image_normalized_file)
            PIL.Image.fromarray(recon_image_norm[:,:,[2,1,0]]).save(recon_image_normalized_file)
            
            # break

    print('All done')

    return output_dir


# Functions ##################################################################

def image_preprocess(img, image_mean=np.float32([104, 117, 123])):
    '''convert to Caffe's input image layout'''
    return np.float32(np.transpose(img, (2, 0, 1))[::-1]) - np.reshape(image_mean, (3, 1, 1))


def image_deprocess(img, image_mean=np.float32([104, 117, 123])):
    '''convert from Caffe's input image layout'''
    return np.dstack((img + np.reshape(image_mean, (3, 1, 1)))[::-1])


# Entry point ################################################################

# SET PARAMETERS ::::::::::
decoded_feat_dir_main = '../../decoded/gan_mod/'
output_dir_main = '../../decoded/gan_recon_img/'
layerType_list = ['all_layers']
subject_list = ['subj01']
roi = 'streams'
nIter_recon = 200

# SET PARAMETERS ::::::::::
for layerType in layerType_list:

    if layerType == 'all_layers':
        layers = ['conv1_2', 'conv2_2', 'conv3_4', 'conv4_4', 'conv5_4', 'fc6', 'fc7', 'fc8']

    output_dir = os.path.join(output_dir_main, layerType)
    recon_icnn_image_vgg19_dgn_relu7gen_dg(\
        layers, decoded_feat_dir_main, output_dir=output_dir,\
        subjects=subject_list, rois=[roi], n_iter=nIter_recon, device='cuda:0')
