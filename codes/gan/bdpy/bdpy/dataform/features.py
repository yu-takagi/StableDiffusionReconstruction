'''DNN features class


This file is a part of BdPy.
'''


from __future__ import print_function


__all__ = ['Features', 'DecodedFeatures']


import os
import glob
import sqlite3
import pickle
import warnings

import numpy as np
import scipy.io as sio
import hdf5storage


class Features(object):
    '''DNN features class.

    Parameters
    ----------
    dpath: str or list
       (List of) DNN feature directory(ies)
    ext: str
        DNN feature file extension (default: mat)

    Attributes
    ----------
    labels: list
       List of stimulus labels
    index: list
       List of stimulus index (one-based)
    layers: list
       List of DNN layers
    '''

    def __init__(self, dpath=[], ext='mat', feature_index=None):
        if type(dpath) != list:
            dpath = [dpath]
        self.__dpath = dpath
        self.__feat_index_table = feature_index

        self.__feature_file_table = {} # Stimulus feature file tables
        self.__labels = []             # Stimulus labels
        self.__index = []              # Stimulus index (one-based)
        self.__feature_index = []      # Feature (unit) index
        self.__layers = []             # DNN layers
        self.__collect_feature_files(ext=ext)

        self.__c_feature_name = None  # Loaded layer
        self.__features = None        # Loaded features
        self.__feature_index = None   # Indexes of loaded features

        if self.__feat_index_table is not None:
            if not os.path.exists(self.__feat_index_table):
                raise RuntimeError('%s do not exist' % self.__feat_index_table)
            self.__feat_index_table = hdf5storage.loadmat(self.__feat_index_table)['index']

        self.__statistics = {}
        for fdir in self.__dpath:
            stat_file = os.path.join(fdir, 'statistics.pkl')
            if os.path.exists(stat_file):
                with open(stat_file, 'rb') as f:
                    feat_stat = pickle.load(f)
                self.__statistics.update(feat_stat)

    @property
    def labels(self):
        return self.__labels

    @property
    def index(self):
        return self.__index

    @property
    def layers(self):
        return self.__layers

    @property
    def feature_index(self):
        return self.__feature_index

    def get(self, layer):
        '''Return features in `layer`.

        Parameters
        ----------
        layer: str
            DNN layer

        Returns
        -------
        numpy.ndarray, shape=(n_samples, shape_layers)
            DNN features
        '''

        return self.get_features(layer)

    def statistic(self, statistic='mean', layer=None):

        if statistic == 'std':
            statistic = 'std, ddof=1'

        k = (statistic, layer)
        if k in self.__statistics:
            s = self.__statistics[k]
        else:
            f = self.get(layer)

            if statistic == 'mean':
                s = np.mean(f, axis=0)[np.newaxis, :]
            elif statistic == 'std, ddof=1':
                s = np.std(f, axis=0, ddof=1)[np.newaxis, :]
            elif statistic == 'std, ddof=0':
                s = np.std(f, axis=0, ddof=0)[np.newaxis, :]
            else:
                raise ValueError('Unknown statistics: {}'.format(statistic))

            self.__statistics.update({k: s})

        if self.__feat_index_table is not None:
            # Select features by index
            self.__feature_index = self.__feat_index_table[layer]
            n_sample = self.__features.shape[0]
            n_feat = np.array(self.__features.shape[1:]).prod()

            s = s.reshape([n_sample, n_feat], order='C')[:, self.__feature_index]

        return s

    def get_features(self, layer):
        '''Return features in `layer`.

        Parameters
        ----------
        layer: str
            DNN layer

        Returns
        -------
        numpy.ndarray, shape=(n_samples, shape_layers)
            DNN features
        '''

        if layer == self.__c_feature_name:
            return self.__features

        try:
            self.__features = np.vstack(
                [sio.loadmat(self.__feature_file_table[layer][label])['feat']
                 for label in self.__labels]
            )
        except NotImplementedError:
            self.__features = np.vstack(
                [hdf5storage.loadmat(self.__feature_file_table[layer][label])['feat']
                 for label in self.__labels]
            )

        self.__c_feature_name = layer

        if self.__feat_index_table is not None:
            # Select features by index
            self.__feature_index = self.__feat_index_table[layer]
            n_sample = self.__features.shape[0]
            n_feat = np.array(self.__features.shape[1:]).prod()

            self.__features = self.__features.reshape([n_sample, n_feat], order='C')[:, self.__feature_index]

        return self.__features

    def save_feature_index(self, fname):
        '''Save feature indexes in `fname`'''
        if len(self.__feature_index) == 0:
            raise RuntimeError('No feature index specified')

        hdf5storage.savemat(fname,
                            {'index': self.__feature_index},
                            format='7.3', oned_as='column',
                            store_python_metadata=True)

    def __collect_feature_files(self, ext='mat'):
        dpath_lst = self.__dpath

        # List-up layers and stimulus labels
        label_dir = {}
        for dpath in dpath_lst:
            # List-up layers
            self.__layers = self.__get_layers(dpath)

            # List-up stimulus labels
            labels_in_dir = self.__get_labels(dpath, self.__layers, ext=ext)
            label_dir.update({label: dpath for label in labels_in_dir})
            self.__labels += labels_in_dir

        self.__index = np.arange(len(self.__labels)) + 1

        # List-up feature files
        for lay in self.__layers:
            self.__feature_file_table.update(
                {
                    lay:
                    {
                        label:
                         os.path.join(label_dir[label], lay, label + '.' + ext)
                         for label in self.__labels
                    }
                })

        return None

    def __get_layers(self, dpath):
        layers = sorted([d for d in os.listdir(dpath) if os.path.isdir(os.path.join(dpath, d))])
        if self.__layers and (layers != self.__layers):
            raise RuntimeError('Invalid layers in %s' % dpath)
        return layers

    def __get_labels(self, dpath, layers, ext='mat'):
        labels = []
        for lay in layers:
            lay_dir = os.path.join(dpath, lay)
            lay_dir = lay_dir.replace('[', '[[]') # Use glob.escape for Python 3.4 or later
            files = glob.glob(os.path.join(lay_dir, '*.' + ext))
            labels_t = sorted([os.path.splitext(os.path.basename(f))[0] for f in files])
            if not labels:
                labels = labels_t
            else:
                if labels != labels_t:
                    raise RuntimeError('Invalid feature file in %s ' % dpath)
        return labels


class DecodedFeatures(object):
    '''Decoded features class.

    Parameters
    ----------
    path: str
       Path to the decoded feature directory
    '''

    def __init__(self, path=None, keys=None, file_ext='mat', file_key='feat', squeeze=False):

        self.__path = path          # Path to decoded feature directory
        self.__keys = keys          # Keys
        self.__file_ext = file_ext  # Decoded feature file extension
        self.__file_key = file_key  # Decoded feature data key (FIXME)
        self.__squeeze = squeeze    # Whether squeeze the output array or not

        if self.__path is not None:
            self.__db = self.__parse_dir(self.__path, self.__keys)
        else:
            self.__db = self.__init_db(self.__keys)

        stat_file = os.path.join(self.__path, 'statistics.pkl')

        if os.path.exists(stat_file):
            with open(stat_file, 'rb') as f:
                self.__statistics = pickle.load(f)
        else:
            self.__statistics = {}

    @property
    def layers(self):
        return self.__db.get_available_values('layer')

    @property
    def subjects(self):
        return self.__db.get_available_values('subject')

    @property
    def rois(self):
        return self.__db.get_available_values('roi')

    @property
    def folds(self):
        return self.__db.get_available_values('fold')

    @property
    def labels(self):
        return self.__db.get_available_values('label')

    @property
    def selected_layer(self):
        return self.__db.get_selected_values('layer')

    @property
    def selected_subject(self):
        return self.__db.get_selected_values('subject')

    @property
    def selected_roi(self):
        return self.__db.get_selected_values('roi')

    @property
    def selected_fold(self):
        return self.__db.get_selected_values('fold')

    @property
    def selected_label(self):
        return self.__db.get_selected_values('label')

    def get(self, layer=None, subject=None, roi=None, fold=None, label=None, image=None):
        '''Returns decoded features as an array.'''

        if image is not None:
            if label is None:
                warnings.warn('`image` will be deprecated.')
                label = image
            else:
                warnings.warn('`image` will be deprecated and overwritten by `label`.')

        files = self.__db.get_file(
            layer=layer,
            subject=subject,
            roi=roi,
            fold=fold,
            label=label
        )

        if len(files) == 0:
            raise RuntimeError('No decoded feature found')

        y = np.vstack(
            [hdf5storage.loadmat(f)[self.__file_key] for f in files]
        )

        if self.__squeeze:
            y = np.squeeze(y)

        return y

    def statistic(self, statistic='mean', layer=None, subject=None, roi=None, fold=None):

        if statistic == 'std':
            statistic = 'std, ddof=1'

        k = (statistic, layer, subject, roi, fold)
        if k in self.__statistics:
            s = self.__statistics[k]
        else:
            f = self.get(layer=layer, subject=subject, roi=roi, fold=fold)

            if statistic == 'mean':
                s = np.mean(f, axis=0)[np.newaxis, :]
            elif statistic == 'std, ddof=1':
                s = np.std(f, axis=0, ddof=1)[np.newaxis, :]
            elif statistic == 'std, ddof=0':
                s = np.std(f, axis=0, ddof=0)[np.newaxis, :]
            else:
                raise ValueError('Unknown statistics: {}'.format(statistic))

            self.__statistics.update({k: s})

        return s

    def __parse_dir(self, path, keys):
        # TODO: refactoring
        if keys is None:
            files = glob.glob(os.path.join(path, '*', '*', '*', '*', 'decoded_features', '*.' + self.__file_ext))
            keys = ['layer', 'subject', 'roi', 'fold', 'label']
            if len(files) == 0:
                files = glob.glob(os.path.join(path, '*', '*', '*', 'decoded_features', '*.' + self.__file_ext))
                keys = ['layer', 'subject', 'roi', 'label']
            if len(files) == 0:
                files = glob.glob(os.path.join(path, '*', '*', '*', '*.' + self.__file_ext))
                keys = ['layer', 'subject', 'roi', 'label']
        elif len(keys) == 4:
            # <layer>/<subject>/<roi>/<label>
            files = glob.glob(os.path.join(path, '*', '*', '*', 'decoded_features', '*.' + self.__file_ext))
            if len(files) == 0:
                files = glob.glob(os.path.join(path, '*', '*', '*', '*.' + self.__file_ext))
            if len(files) == 0:
                raise RuntimeError('Decoded features not found')
        elif len(keys) == 5:
            # <layer>/<subject>/<roi>/<fold>/<label>
            files = glob.glob(os.path.join(path, '*', '*', '*', '*', 'decoded_features', '*.' + self.__file_ext))
        else:
            raise ValueError('Invalid keys')

        if len(files) == 0:
            raise RuntimeError('Decoded features not found')

        print('Found {} decoded features in {}'.format(len(files), self.__path))

        self.__keys = keys

        db = FileDatabase(keys)

        # TODO: performance improvement
        for file in files:
            # FXIME: "decoded_features"
            k = {
                k: os.path.splitext(file.replace('/decoded_features/', '/'))[0].split('/')[i - len(keys)]
                for i, k in enumerate(keys)
            }
            db.add_file(file, **k)

        return db

    def __init_db(self, keys):
        raise NotImplementedError


class FileDatabase(object):
    def __init__(self, keys):
        self.__keys = keys

        self.__res = None

        self.__con = sqlite3.connect(':memory:')
        self.__cursor = self.__con.cursor()

        self.__cursor.execute(
            '''
            CREATE TABLE files (
            {},
            path TEXT,
            UNIQUE ({})
            )
            '''.format(
                (', ').join([s + ' TEXT' for s in keys]),
                (', ').join(keys)
            )
        )

    def add_file(self, path, **kargs):
        key_list = ', '.join(kargs) + ', path'
        val_list = ', '.join(['"{}"'.format(s) for s in kargs.values()]) + ', "{}"'.format(path)
        self.__cursor.execute('INSERT INTO files({}) VALUES ({})'.format(key_list, val_list))

    def get_file(self, **kargs):
        where = ' AND '.join(['{} = "{}"'.format(k, v) for k, v in kargs.items() if k in self.__keys and v is not None])
        self.__cursor.execute('SELECT * FROM files WHERE {}'.format(where))
        self.__res = self.__cursor.fetchall()
        return [a[-1] for a in self.__res]

    def get_available_values(self, key):
        if not key in self.__keys:
            return None
        self.__cursor.execute('SELECT DISTINCT {} FROM files'.format(key))
        return [a[0] for a in self.__cursor.fetchall()]

    def get_selected_values(self, key):
        if not key in self.__keys:
            return None
        return [a[self.__keys.index(key)] for a in self.__res]

    def show(self):
        self.__cursor.execute('SELECT * FROM files')
        print(self.__cursor.fetchall())
