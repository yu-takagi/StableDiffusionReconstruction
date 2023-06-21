'''DataStore class

This file is a part of BdPy.
'''


from __future__ import print_function

import os
import glob
import re

import numpy as np
import scipy.io as sio
import h5py
import hdf5storage


__all__ = ['DataStore', 'DirStore']


class DataStore(object):
    '''Data store class.

    Parameters
    ----------
    dpath : str or list
        Path(s) to data directory(ies).
    file_type : {'mat', 'mat_hdf5'}
        Data file format.
    pattern : str, regular expression pattern
        Regular expression pattern to parse file names (paths).
    extractor : func, optional
        Function to extract data from files.

    Example
    -------

    # Suppose you have mat files in `/path/to/data/dir/`. Each file has name
    # such as `subject1_V1.mat`, and contains variable `data`.

    datastore = DataStore('/path/to/data/dir/',
                          file_type='mat',
                          pattern='.*/(.*?)_(.*?).mat',
                          extractor=lambda x: x['data'])

    dat = datastore.get('subject1', 'V1')

    TODO
    ----
    - Add input checks.
    - Add recursive file search.
    - Add default file name pattern (`pattern`).
    '''

    def __init__(self,
                 dpath=None, file_type=None,
                 pattern=None, extractor=None):
        self.__key_sep = '/'

        if isinstance(dpath, str):
            dpath = [dpath]

        self.root_path = dpath
        self.file_type = file_type
        self.pattern = pattern
        self.extractor = extractor

        self.n_keys = self.__get_key_num(self.pattern)

        self.file_dict = {}

        if self.root_path is not None:
            for p in self.root_path:
                self.__parse_datafiles(p)

    def get(self, *keys):
        '''Get data specified by keys.

        Parameters
        ----------
        keys : str
            Keys to specify data.

        Returns
        -------
        All variables in a file (dict) or extracted data.
        '''

        fpath = self.__get_file_from_keys(keys)
        dat = self.__load_data(fpath, self.extractor)

        return dat

    def __get_file_from_keys(self, keys_lst):
        '''Get file path specified by `keys_lst`.'''
        key = self.__key_sep.join(keys_lst)
        return self.file_dict[key]

    def __load_data(self, fpath, extractor):
        '''Load data in `fpath`.'''
        print('Loading ' + fpath)

        if self.file_type is None:
            raise RuntimeError('File type unspecified')
        if self.file_type == 'mat':
            dat = self.__load_data_mat(fpath, extractor)
        elif self.file_type == 'mat_hdf5':
            dat = self.__load_data_mat_hdf5(fpath, extractor)
        else:
            raise ValueError('Unknown file type: %s' % self.file_type)

        return dat

    def __load_data_mat(self, fpath, extractor):
        if extractor is None:
            return sio.loadmat(fpath)
        else:
            return extractor(sio.loadmat(fpath))

    def __load_data_mat_hdf5(self, fpath, extractor):
        if extractor is None:
            with h5py.File(fpath, 'r') as f:
                return f
        else:
            with h5py.File(fpath, 'r') as f:
                return extractor(f)

    def __get_key_num(self, pat):
        '''Return no. of keys.'''
        n_keys = len(re.findall('\(.*?\)', pat))
        return n_keys

    def __parse_datafiles(self, dpath):
        '''Parse files in `dpath`.'''

        print('Searching %s' % dpath)

        if not os.path.isdir(dpath):
            raise ValueError('Invalid directory path: %s' % dpath)

        if self.file_type is None:
            ext = ''
        elif self.file_type == 'mat':
            ext = '.mat'
        elif self.file_type == 'mat_hdf5':
            ext = '.mat'
        else:
            raise ValueError('Unknown file type: %s' % self.file_type)

        parse = re.compile(self.pattern)

        for f in glob.glob(dpath + '/*' + ext):
            m = parse.match(f)
            if m:
                key_list = [m.group(i) for i in range(1, self.n_keys + 1)]
                key = self.__key_sep.join(key_list)
                self.file_dict.update({key: f})


class DirStore(object):
    '''Directory-based data store class.

    Parameters
    ----------
    dpath : str
        Path to data directory.
    dirs_pattern : list
        Directory structure definition.
        For example, dirs_pattern = ['layer', 'subject', 'roi'] defines
        following data structure:

            <dpath>/<layer>/<subject>/<roi>

    file_pattern : str
        File name pattern definition (e.g., <image>.mat).
    variable : str
        Variable name in the data file.
    squeeze : bool
        Squeeze the data if True.


    '''

    def __init__(self, dpath,
                 dirs_pattern=[],
                 file_pattern=None,
                 variable=None,
                 squeeze=False):
        self.__dpath = dpath
        self._dpath = dpath
        self.__dirs_pattern = dirs_pattern
        self.__file_pattern = file_pattern
        self.__variable = variable
        self.__squeeze = squeeze

        self._file_names = []

    def get(self, **kargs):
        '''Returns data specified by kargs.

        Example
        -------

        Files are organized as below:

            <dpath>/<layer>/<subject>/<roi>/<image>.mat

        Then,

            ds = DirStore('./data/dir',
                          dirs_pattern=['layer', 'subject', 'roi'],
                          file_pattern='<image>.mat',
                          variable='feat')
            data = ds.get(layer='conv1', subject='TH', roi='VC', image='Image_001')

        the above code reads ./data/dir/conv1/subject/TH/VC/Image_001.mat and
        returns variable 'feat' in the file.
        '''

        # Sub-directories
        subdir_path = os.path.join(*[kargs[p] for p in self.__dirs_pattern])

        # File name
        file_name = self.__file_pattern
        match = re.findall('<(.*?)>', file_name)
        replace_dict = {}
        for m in match:
            if m in kargs:
                replace_dict.update({'<' + m + '>': kargs[m]})

        if not replace_dict:
            match = re.findall('<.*>(.*)', file_name)
            # FXIME
            file_path = os.path.join(self.__dpath, subdir_path, '*' + match[0])
        else:
            for k, v in replace_dict.items():
                file_name = file_name.replace(k, v)
                file_path = os.path.join(self.__dpath, subdir_path, file_name)

        # Get files
        files = sorted(glob.glob(file_path))

        if len(files) == 0:
            raise RuntimeError('File not found: %s' % file_path)
        elif len(files) >= 2:
            # FIXME
            dat = np.vstack([
                self.__load_feature(f)
                for f in files
            ])
            self._file_names = [
                os.path.splitext(os.path.basename(f))[0]
                for f in files
            ]
        else:
            dat = self.__load_feature(files[0])

        return dat

    def __load_feature(self, fpath):
        r = hdf5storage.loadmat(fpath)[self.__variable]
        if self.__squeeze:
            r = np.squeeze(r)
        return r
