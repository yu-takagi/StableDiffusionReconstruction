'''BrainDecoderToolbox2/BdPy data class

This file is a part of BdPy


API list
--------

- Data modification
    - add
    - update
    - add_metadata
    - rename_metadata
    - set_metadatadescription
- Data access
    - select
    - get
    - get_metadata
    - show_metadata
- File I/O
    - load
    - save
'''


__all__ = ['BData']


import os
import sys
import warnings
import time
import datetime
import inspect
import re

import h5py
import numpy as np
import scipy.io as sio

from .metadata import MetaData
from .featureselector import FeatureSelector


# BData class ##########################################################

class BData(object):
    '''BrainDecoderToolbox2/BdPy data class

    The instance of class `BData` contains `dataset` and `metadata` as instance
    variables.

    Parameters
    ----------
    file_name : str, optional
        File which contains BData (default: None)
    file_type : {'Matlab', 'HDF5', 'None'}, optional
        File type (default: None)

    If `file_name` was not given, BData.__init__() creates an empty
    dataset and metadata.

    Attributes
    ----------
    dataset : numpy array (dtype=float)
        Dataset array
    metadata : metadata object
        Meta-data object
    '''


    def __init__(self, file_name=None, file_type=None):
        self.__dataset = np.ndarray((0, 0), dtype=float)
        self.__metadata = MetaData()
        self.__header = {}
        self.__vmap = {}

        if file_name is not None:
            self.load(file_name, file_type)

    # Properties -------------------------------------------------------

    # dataset
    @property
    def dataset(self):
        return self.__dataset

    @dataset.setter
    def dataset(self, value):
        self.__dataset = value

    # metadata
    @property
    def metadata(self):
        return self.__metadata

    @metadata.setter
    def metadata(self, value):
        self.__metadata = value

    # header
    @property
    def header(self):
        return self.__header

    # dataSet
    @property
    def dataSet(self):
        return self.__dataset

    @dataSet.setter
    def dataSet(self, value):
        self.__dataset = value

    # metaData
    @property
    def metaData(self):
        return self.__metadata

    @metaData.setter
    def metaData(self, value):
        self.__metadata = value

    # Misc -------------------------------------------------------------

    def __obsoleted_method(alternative):
        '''Decorator for obsoleted functions'''
        def __obsoleted_method_in(func):
            import functools
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                funcname = func.__name__
                warnings.warn("'%s' is obsoleted and kept for compatibility. Use '%s' instead." % (funcname, alternative),
                              UserWarning, stacklevel=2)
                return func(*args, **kwargs)
            return wrapper
        return __obsoleted_method_in


    # Data modification ------------------------------------------------

    def add(self, x, name):
        '''Add `x` to dataset as `name`.`

        Parameters
        ----------
        x : array
            Data matrix to be added in dataset
        name : str
            Name of the data `x`

        Returns
        -------
        None
        '''

        if x.ndim == 1:
            x = x[:, np.newaxis]

        colnum_has = self.dataset.shape[1] # Num of existing columns in 'dataset'
        colnum_add = x.shape[1]            # Num of columns to be added

        # Add 'x' to dataset
        if not self.dataset.size:
            self.dataset = x
        else:
            # TODO: Add size check of 'x' and 'self.dataset'
            self.dataset = np.hstack((self.dataset, x))

        # Add new metadata
        column_description = '1 = %s' % name
        column_value = [np.nan for _ in range(colnum_has)] + [1 for _ in range(colnum_add)]

        self.metadata.set(name, column_value, column_description,
                          lambda x, y: np.hstack((y[:colnum_has], x[-colnum_add:])))


    @__obsoleted_method('add')
    def add_dataset(self, x, attribute_key):
        '''Add `x` to dataset with attribute meta-data key `attribute_key`

        Parameters
        ----------
        x : array
            Data matrix to be added in dataset
        attribute_key : str
            Key of attribute meta-data, which specifies the columns containing `x`

        Returns
        -------
        None
        '''
        return self.add(x, attribute_key)


    def update(self, key, dat):
        '''Update dataset

        Parameters
        ----------
        key : str
           Name of columns to be updated
        dat : array_like
           New data array

        Returns
        -------
        None
        '''
        mdind = [a == 1 for a in self.get_metadata(key)]
        self.dataset[:, np.array(mdind)] = dat


    def add_metadata(self, key, value, description='', where=None, attribute=None):
        '''Add meta-data with `key`, `description`, and `value` to metadata

        Parameters
        ----------
        key : str
            Meta-data key
        value : array
            Meta-data array
        description : str, optional
            Meta-data description
        where : str, optional
            Meta-data key masking the columns in the dataset

        Returns
        -------
        None
        '''

        # TODO: Add attribute specifying
        # TODO: Add size check of metadata/value

        if attribute is not None:
            warnings.warn("Keyword argument 'attribute' is obsoleted and kept for compatibility. Use 'where' instead.",
                          UserWarning, stacklevel=2)
            if where is not None:
                warnings.warn("Value of 'attribute' is overridden by 'where'.",
                              UserWarning, stacklevel=2)
            else:
                where = attribute

        if where is not None:
            attr_ind = self.metadata.get(where, 'value') == 1
            add_value = np.array([np.nan for _ in range(self.metadata.get_value_len())])
            add_value[attr_ind] = value
        else:
            add_value = value

        self.metadata.set(key, add_value, description)


    def merge_metadata(self, key, sources, description='', where=None, method='logical_or'):
        '''Merage metadata rows.'''

        if not method == 'logical_or':
            raise NotImplementedError('Only `logical_or` is implemented')

        if where is None:
            raise ValueError('You need to specify `where`.')

        mdv_lst = [self.get_metadata(s, where=where) for s in sources]
        mdv_new = np.nansum(np.vstack(mdv_lst), axis=0)
        mdv_new[mdv_new > 1] = 1

        self.add_metadata(key, mdv_new, description, where=where)


    def rename_metadata(self, key_old, key_new):
        '''Rename meta-data key

        Parameters
        ----------
        key_old, key_new : str
            Old and new meta-data keys

        Returns
        -------
        None
        '''
        self.metadata.key[self.metadata.key.index(key_old)] = key_new
        return None


    def set_metadatadescription(self, key, description):
        '''Set description of metadata specified by `key`

        Parameters
        ----------
        key : str
            Meta-data key
        description : str
            Meta-data description

        Returns
        -------
        None
        '''

        self.metadata.set(key, None, description,
                          lambda x, y: y)


    @__obsoleted_method('set_metadatadescription')
    def edit_metadatadescription(self, metakey, description):
        '''Set description of metadata specified by `key`

        Parameters
        ----------
        key : str
            Meta-data key
        description : str
            Meta-data description

        Returns
        -------
        None
        '''
        self.set_metadatadescription(metakey, description)

    def update_header(self, header):
        '''Update header.'''
        self.__header.update(header)

    def applyfunc(self, func, where=None, **kargs):
        '''Apply `func` to the dataset.'''

        if where is None:
            # FIXME
            fout = func(self.dataset, **kargs)

            if isinstance(fout, tuple):
                self.dataset = fout[0]
            else:
                self.dataset = fout
        else:
            # FIXME
            if not isinstance(where, list):
                where = [where]

            data_selector = '|'.join([w + ' = 1' for w in where])

            x, x_ind = self.select(data_selector, return_index=True)

            fout = func(x, **kargs)

            if isinstance(fout, tuple):
                # Index mapping
                ind_map = fout[1]

                ds = np.zeros((len(ind_map), self.dataset.shape[1]))

                index = np.zeros(self.dataset.shape[1], dtype=bool)
                index[x_ind] = True

                #import pdb; pdb.set_trace()

                ds[:, index] = fout[0]
                ds[:, ~index] = self.dataset[np.ix_(ind_map, ~index)]

                self.dataset = ds
            else:
                # No index mapping
                self.dataset[:, x_ind] = fout

        return self

    # Data access ------------------------------------------------------

    def select(self, condition, return_index=False, verbose=True):
        '''Select data (columns) from dataset.

        Parameters
        ----------
        condition : str
            Condition specifying columns.
        retrun_index : bool, optional
            If True, return index of selected columns (default: False).
        verbose : bool, optional
            If True, display verbose messages (default: True).

        Returns
        -------
        array-like
            Selected data
        list, optional
            Selected index

        Note
        ----
        The following operators are acceptable in `condition`.

        - | (or)
        - & (and)
        - = (equal)
        - @ (conditional)
        '''

        expr_rpn = FeatureSelector(condition).rpn

        stack = []
        buf_sel = []

        for i in expr_rpn:
            if i == '=':
                r = stack.pop()
                l = stack.pop()

                stack.append(np.array([n == r for n in l], dtype=bool))

            elif i == 'top':
                # Dirty solution

                # Need fix on handling 'None'

                n = int(stack.pop()) # Num of elements to be selected
                v = stack.pop()

                order = self.__get_order(v)

                stack.append(order)
                buf_sel.append(n)

            elif i in ['|', '&', '-']:
                r = stack.pop()
                l = stack.pop()

                if r.dtype != 'bool':
                    # 'r' should be an order vector
                    num_sel = buf_sel.pop()
                    r = self.__get_top_elm_from_order(r, num_sel)
                    #r = np.array([ n < num_sel for n in r ], dtype = bool)

                if l.dtype != 'bool':
                    # 'l' should be an order vector
                    num_sel = buf_sel.pop()
                    l = self.__get_top_elm_from_order(l, num_sel)
                    #l = np.array([ n < num_sel for n in l ], dtype = bool)

                if i == '|':
                    result = np.logical_or(l, r)
                elif i == '&':
                    result = np.logical_and(l, r)
                elif i == '-':
                    result = np.logical_and(l, np.logical_not(r))

                stack.append(result)

            elif i == '@':
                # FIXME
                # In the current version, the right term of '@' is assumed to
                # be a boolean, and the left is to be an order vector.

                r = stack.pop() # Boolean
                l = stack.pop() # Float

                l[~r] = np.inf

                selind = self.__get_top_elm_from_order(l, buf_sel.pop())

                stack.append(np.array(selind))

            else:
                if isinstance(i, str):
                    if i.isdigit():
                        # 'i' should be a criteria value
                        i = float(i)
                    else:
                        # 'i' should be a meta-data key
                        i = self.__metadata_key_to_bool_vector(i)

                stack.append(i)

        selected_index = stack.pop()

        # If buf_sel still has an element, `select_index` should be an order vector.
        # Select N elements based on the order vector.
        if buf_sel:
            num_sel = buf_sel.pop()
            selected_index = [n < num_sel for n in selected_index]

        # Very dirty solution
        selected_index = np.array(selected_index) == True

        if return_index:
            return self.dataset[:, np.array(selected_index)], selected_index
        else:
            return self.dataset[:, np.array(selected_index)]


    @__obsoleted_method('select')
    def select_dataset(self, condition, return_index=False, verbose=True):
        '''Select data (columns) from dataset.

        Parameters
        ----------
        condition : str
            Condition specifying columns.
        retrun_index : bool, optional
            If True, return index of selected columns (default: False).
        verbose : bool, optional
            If True, display verbose messages (default: True).

        Returns
        -------
        array-like
            Selected data
        list, optional
            Selected index

        Note
        ----
        The following operators are acceptable in `condition`.

        - | (or)
        - & (and)
        - = (equal)
        - @ (conditional)
        '''
        return self.select(condition, return_index, verbose)


    @__obsoleted_method('select')
    def select_feature(self, condition, return_index=False, verbose=True):
        '''Select data (columns) from dataset.

        Parameters
        ----------
        condition : str
            Condition specifying columns.
        retrun_index : bool, optional
            If True, return index of selected columns (default: False).
        verbose : bool, optional
            If True, display verbose messages (default: True).

        Returns
        -------
        array-like
            Selected data
        list, optional
            Selected index

        Note
        ----
        The following operators are acceptable in `condition`.

        - | (or)
        - & (and)
        - = (equal)
        - @ (conditional)
        '''
        return self.select(condition, return_index, verbose)


    def get(self, key=None):
        '''Get dataset

        When `key` is not given, `get_dataset` returns `dataset`. When `key` is
        given, `get_dataset` returns data specified by `key`
        '''

        if key is None:
            return self.dataset
        else:
            query = '%s = 1' % key
            return self.select(query, return_index=False, verbose=False)


    @__obsoleted_method('get')
    def get_dataset(self, key=None):
        '''Get dataset

        When `key` is not given, `get_dataset` returns `dataset`. When `key` is
        given, `get_dataset` returns data specified by `key`
        '''
        return self.get(key)


    def get_metadata(self, key, where=None):
        '''Get value of meta-data specified by `key`

        Parameters
        ----------
        key : str
            Meta-data key.

        where : str, optional
            Columns which mask meta-data array.

        Returns
        -------
        array-like
        '''

        md = self.metadata.get(key, 'value')

        if where != None:
            # Mask the metadata array with columns specified with `where`
            ind = self.metadata.get(where, 'value') == True
            md = md[ind]

        return md


    def show_metadata(self):
        '''Show all the key and description in metadata'''

        # Get max length
        max_key = max([len(k) for k in self.metadata.key])
        max_desc = max([len(k) for k in self.metadata.description])

        # Disp header
        print('| ' + 'Key' + ' ' * (max_key - 3) + ' | ' + 'Description' + ' ' * (max_desc - 11) + ' |')
        print('|-' + '-' * max_key + '-|-' + '-' * max_desc + '-|')

        # Disp key and description
        for k, d in zip(self.metadata.key, self.metadata. description):
            print('| ' + k + ' ' * (max_key - len(k)) + ' | ' + d + ' ' * (max_desc - len(d)) + ' |')

    # Value-label map --------------------------------------------------------

    def get_labels(self, key):
        '''Get `key` as labels.'''
        if not key in self.__vmap:
            raise ValueError('Key not found in vmap: %s' % key)
        value = self.select(key).flatten()
        label = []
        for x in value:
            if np.isnan(x):
                v = 'n/a'
            else:
                v = self.__vmap[key][x]
            label.append(v)
        return label

    def get_label(self, key):
        '''Get `key` as labels.'''
        return self.get_labels(key)

    def get_vmap(self, key):
        '''Returns vmap of `key`.'''
        if key in self.__vmap:
            return self.__vmap[key]
        else:
            raise ValueError('%s not found in vmap' % key)

    def get_vmap_keys(self):
        return self.__vmap.keys()

    def add_vmap(self, key, vmap):
        '''Add vmap.'''
        if not key in self.__metadata.key:
            raise ValueError('%s not found in metadata.' % key)

        if type(vmap) is not dict:
            raise TypeError('`vmap` should be a dictionary.')
        for k in vmap.keys():
            if type(k) is str:
                raise TypeError('Keys of `vmap` should be numerical.')

        if key in self.__vmap:
            # Check vmap consistency
            if self.__check_vmap_consistency(vmap, self.__vmap[key]):
                vmap.update(self.__vmap[key])
                vmap_add = self.__get_act_vmap(key, vmap)
                self.__vmap[key].update(vmap_add)
            else:
                raise ValueError('Invalid vmap: labels are inconsistent between old and new vmaps.')
        else:
            vmap_add = self.__get_act_vmap(key, vmap)
            self.__vmap.update({key: vmap_add})

        return None

    def __get_act_vmap(self, key, vmap):
        values = np.unique(self.get(key))
        try:
            vmap_add = {}
            for val in values:
                if np.isnan(val):
                    continue
                vmap_add.update({val: vmap[val]})
        except KeyError:
            raise ValueError('Invalid vmap: label for %f not found.' % val)
        return vmap_add

    def __check_vmap_consistency(self, vmap_new, vmap_old):
        for key in vmap_new.keys():
            if not key in vmap_old:
                continue
            if vmap_old[key] != vmap_new[key]:
                print('Inconsistent label:')
                print('    Key: %f' % key)
                print('    Old label: %s' % vmap_old[key])
                print('    New label: %s' % vmap_new[key])
                return False
        return True

    # File I/O---------------------------------------------------------

    def load(self, load_filename, load_type=None):
        '''Load 'dataset' and 'metadata' from a given file'''

        if load_type is None:
            load_type = self.__get_filetype(load_filename)

        if load_type == "Matlab":
            self.__load_mat(load_filename)
        elif load_type == "HDF5":
            self.__load_h5(load_filename)
        else:
            raise ValueError("Unknown file type: %s" % (load_type))


    def save(self, file_name, file_type=None):
        '''Save 'dataset' and 'metadata' to a file'''

        # Store data creation information
        t_now = time.time()
        t_now_str = datetime.datetime.fromtimestamp(t_now).strftime('%Y-%m-%d %H:%M:%S')

        callstack = []
        callstack_code = []
        f = inspect.currentframe()
        while True:
            f = f.f_back
            if f is None: break
            fname = os.path.abspath(f.f_code.co_filename)
            fline = f.f_lineno
            callstack.append('%s:%d' % (fname, fline))
            if os.path.exists(fname):
                with open(fname, 'r') as fl:
                    fcode = fl.read()
            else:
                fcode = ''
            callstack_code.append(fcode)

        self.__header.update({'ctime': t_now_str,
                              'ctime_epoch': t_now,
                              'callstack': callstack,
                              'callstack_code': callstack_code})

        if file_type is None:
            file_type = self.__get_filetype(file_name)

        if file_type == "Matlab":
            raise RuntimeError('Saving BData as a mat file is no longer supported. Please save the data as HDF5 (.h5).')
        elif file_type == "HDF5":
            self.__save_h5(file_name, header=self.__header)

        else:
            raise ValueError("Unknown file type: %s" % (file_type))


    # Private methods --------------------------------------------------

    def __metadata_key_to_bool_vector(self, key):
        '''Convert meta-dat key(s) to boolean vector.'''
        key_esc = re.escape(key).replace('\*', '.*') + '$'  # `fullmatch` is available in Python >= 3.4
        keys = [k for k in self.metadata.key if re.match(key_esc, k)]
        if len(keys) == 0:
            raise RuntimeError('Meta-data %s not found' % key)
        vals = np.vstack([
            self.get_metadata(k)
            for k in keys])
        vals = (vals == 1)
        vec = np.sum(vals, axis=0).astype(bool)
        return vec

    def __get_order(self, v, sort_order='descend'):

        # 'np.nan' comes to the last of an acending series, and thus the top of a decending series.
        # To avoid that, convert 'np.nan' to -Inf.
        v[np.isnan(v)] = -np.inf

        sorted_index = np.argsort(v)[::-1] # Decending order
        order = np.arange(len(v))
        for i, x in enumerate(sorted_index):
            order[x] = i

        return np.array(order, dtype=float)


    def __get_top_elm_from_order(self, order, n):
        '''Get a boolean index of top `n` elements from `order`'''
        sorted_index = np.argsort(order)
        for i, x in enumerate(sorted_index):
            order[x] = i

        index = np.array([r < n for r in order], dtype=bool)

        return index


    def __save_h5(self, file_name, header=None):
        '''Save data in HDF5 format (*.h5)'''
        with h5py.File(file_name, 'w') as h5file:
            # dataset
            h5file.create_dataset('/dataset', data=self.dataset)

            # metadata
            md_keys = self.metadata.key
            md_desc = self.metadata.description
            md_vals = self.metadata.value

            h5file.create_group('/metadata')
            h5file.create_dataset('/metadata/key', data=[self.__to_bytes(x) for x in md_keys])
            h5file.create_dataset('/metadata/description', data=[self.__to_bytes(x) for x in md_desc])
            h5file.create_dataset('/metadata/value', data=md_vals)

            # header
            if header is not None:
                h5file.create_group('/header')
                for k, v in header.items():
                    if isinstance(v, list):
                        h5file.create_dataset('/header/' + k, data=[self.__to_bytes(x) for x in v])
                    else:
                        h5file.create_dataset('/header/' + k, data=self.__to_bytes(v)) # FIXME: save unicode str as is

            # vmap
            h5file.create_group('/vmap')
            for mk, vm in self.__vmap.items():
                h5file.create_group('/vmap/' + mk)
                for k, v in vm.items():
                    h5file.create_dataset('/vmap/' + mk + '/' + str(k), data=self.__to_bytes(v)) # FIXME: save unicode str as is

    def __load_mat(self, load_filename):
        '''Load dataset and metadata from Matlab file'''

        dat = sio.loadmat(load_filename)

        if 'metaData' in dat:
            md_keys = [str(i[0]).strip() for i in np.asarray(dat["metaData"]['key'][0, 0])[0].tolist()]
            md_descs = [str(i[0]).strip() for i in np.asarray(dat["metaData"]['description'][0, 0])[0].tolist()]
            md_values = np.asarray(dat["metaData"]['value'][0, 0])
        else:
            md_keys = [str(i[0]).strip() for i in np.asarray(dat["metadata"]['key'][0, 0])[0].tolist()]
            md_descs = [str(i[0]).strip() for i in np.asarray(dat["metadata"]['description'][0, 0])[0].tolist()]
            md_values = np.asarray(dat["metadata"]['value'][0, 0])

        if 'dataSet' in dat:
            self.dataset = np.asarray(dat["dataSet"])
        else:
            self.dataset = np.asarray(dat["dataset"])

        if 'header' in dat:
            self.__header = dat['header']

        self.__metadata.key = md_keys
        self.__metadata.value = md_values
        self.__metadata.description = md_descs


    def __load_h5(self, load_filename):
        '''Load dataset and metadata from HDF5 file'''

        dat = h5py.File(load_filename, 'r')

        if 'metaData' in dat:
            md_keys = [self.__to_unicode(x) for x in dat["metaData"]['key'][:].tolist()]
            md_descs = [self.__to_unicode(x) for x in dat["metaData"]['description'][:].tolist()]
            md_values = np.asarray(dat["metaData"]['value'], dtype=np.float)
        else:
            md_keys = [self.__to_unicode(x) for x in dat["metadata"]['key'][:].tolist()]
            md_descs = [self.__to_unicode(x) for x in dat["metadata"]['description'][:].tolist()]
            md_values = np.asarray(dat["metadata"]['value'], dtype=np.float)

        if 'dataSet' in dat:
            self.dataset = np.asarray(dat["dataSet"], dtype=np.float)
        else:
            self.dataset = np.asarray(dat["dataset"], dtype=np.float)

        if 'header' in dat:
            for k, v in dat['header'].items():
                k = self.__to_unicode(k)
                if isinstance(v[()], np.ndarray):
                    v = [self.__to_unicode(x) for x in v[()]]
                else:
                    v = self.__to_unicode(v[()])
                self.__header.update({k: v})

        if 'vmap' in dat:
            for mk in dat['vmap'].keys():
                vmap = {}
                for k in dat['vmap'][mk].keys():
                    vmap.update({float(k): self.__to_unicode(dat['vmap'][mk][k][()])})
                # TODO: fix this dirty solution
                if sys.version_info.major == 2:
                    mk = mk.encode('utf-8')
                self.__vmap.update({mk: vmap})

        self.__metadata.key = md_keys
        self.__metadata.value = md_values
        self.__metadata.description = md_descs

    def __to_unicode(self, s):
        '''Convert s (bytes) to unicode str.'''
        if sys.version_info.major == 3:
            if isinstance(s, bytes):
                return s.decode('utf-8')
        return s

    def __to_bytes(self, s):
        '''Convert s (unicode str) to bytes.'''
        if sys.version_info.major == 3:
            if isinstance(s, str):
                return s.encode('utf-8')
        return s

    def __get_filetype(self, file_name):
        '''Return the type of `file_name` based on the file extension'''

        _, ext = os.path.splitext(file_name)

        if ext == ".mat":
            file_type = "Matlab"
        elif ext == ".h5":
            file_type = "HDF5"
        else:
            raise ValueError("Unknown file extension: %s" % (ext))

        return file_type
