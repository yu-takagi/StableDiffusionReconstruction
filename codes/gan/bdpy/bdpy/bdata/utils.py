'''Utility functions for BData'''


import copy
import sys

import numpy as np

from .bdata import BData


def vstack(bdata_list, successive=[], metadata_merge='strict', ignore_metadata_description=False):
    '''Concatenate datasets vertically.

    Currently, `concat_dataset` does not validate the consistency of meta-data
    among data.

    Parameters
    ----------
    bdata_list : list of BData
        Data to be concatenated
    successsive : list, optional
        Sucessive columns. The values of columns specified here are inherited
        from the preceding data.
    metadata_merge : str, optional
        Meta-data merge strategy ('strict' or 'minimal'; default: strict).
        'strict' requires that concatenated datasets share exactly the same meta-data.
        'minimal' keeps meta-data only shared across the concatenated datasets.
    ignore_metadata_description : bool
        Ignore meta-data description when merging the datasets (default: False).

    Returns
    -------
    dat : BData
        Concatenated data

    Example
    -------

        data = vstack([data0, data1, data2], successive=['Session', 'Run', 'Block'])
    '''

    suc_cols = {s : 0 for s in successive}

    dat = BData()  # Concatenated BData

    for ds in bdata_list:
        ds_copy = copy.deepcopy(ds)

        # Update sucessive columns
        for s in successive:
            v = ds_copy.select(s)
            v += suc_cols[s]
            ds_copy.update(s, v)

        # Concatenate BDatas
        if dat.dataset.shape[0] == 0:
            # Create new BData
            dat.dataset = ds_copy.dataset
            dat.metadata = ds_copy.metadata

            # vmap
            vmap_keys = ds_copy.get_vmap_keys()
            for vk in vmap_keys:
                if sys.version_info.major == 2:
                    dat.add_vmap(vk.encode(), ds_copy.get_vmap(vk.encode()))
                else:
                    dat.add_vmap(vk, ds_copy.get_vmap(vk))
        else:
            # Check metadata consistency
            if metadata_merge == 'strict':
                if not metadata_equal(dat, ds_copy):
                    raise ValueError('Inconsistent meta-data.')
            elif metadata_merge == 'minimal':
                # Only meta-data shared across BDatas are kept.
                shared_mkeys = sorted(list(set(dat.metadata.key) & set(ds_copy.metadata.key)))
                shared_mdesc = []
                shared_mvalue_lst = []
                for mkey in shared_mkeys:
                    d0_desc, d0_value = dat.metadata.get(mkey, 'description'), dat.metadata.get(mkey, 'value')
                    d1_desc, d1_value = ds_copy.metadata.get(mkey, 'description'), ds_copy.metadata.get(mkey, 'value')

                    if not ignore_metadata_description and not d0_desc == d1_desc:
                        raise ValueError('Inconsistent meta-data description (%s)' % mkey)
                    try:
                        np.testing.assert_equal(d0_value, d1_value)
                    except AssertionError:
                        raise ValueError('Inconsistent meta-data value (%s)' % mkey)
                    shared_mdesc.append(d0_desc)
                    shared_mvalue_lst.append(d0_value)
                shared_mvalue = np.vstack(shared_mvalue_lst)

                dat.metadata.key = shared_mkeys
                dat.metadata.description = shared_mdesc
                dat.metadata.value = shared_mvalue
            else:
                raise ValueError('Unknown meta-data merge strategy: %s' % metadata_merge)

            # Concatenate BDatas
            dat.dataset = np.vstack([dat.dataset, ds_copy.dataset])

            # Merge vmap
            vmap_keys = ds_copy.get_vmap_keys()
            for vk in vmap_keys:
                if sys.version_info.major == 2:
                    dat.add_vmap(vk.encode(), ds_copy.get_vmap(vk.encode()))
                else:
                    dat.add_vmap(vk, ds_copy.get_vmap(vk))

        # Update the last values in sucessive columns
        for s in successive:
            v = dat.select(s)
            suc_cols[s] = np.max(v)

    return dat


def concat_dataset(data_list, successive=[]):
    '''Concatenate datasets

    Currently, `concat_dataset` does not validate the consistency of meta-data
    among data.

    Parameters
    ----------
    data_list : list of BData
        Data to be concatenated
    successsive : list, optional
        Sucessive columns. The values of columns specified here are inherited
        from the preceding data.

    Returns
    -------
    dat : BData
        Concatenated data

    Example
    -------

        data = concat_dataset([data0, data1, data2], successive=['Session', 'Run', 'Block'])
    '''

    return vstack(data_list, successive=successive)


def metadata_equal(d0, d1, strict=False):
    '''Check whether `d0` and `d1` share the same meta-data.

    Parameters
    ----------
    d0, d1 : BData
    strict : bool, optional

    Returns
    -------
    bool
    '''

    equal = True

    # Strict check
    if not d0.metadata.key == d1.metadata.key:
        equal = False
    if not d0.metadata.description == d1.metadata.description:
        equal = False
    try:
        np.testing.assert_equal(d0.metadata.value, d1.metadata.value)
    except AssertionError:
        equal = False

    if equal:
        return True

    if strict:
        return False

    # Loose check (ignore the order of meta-data)
    d0_mkeys = sorted(d0.metadata.key)
    d1_mkeys = sorted(d1.metadata.key)
    if not d0_mkeys == d1_mkeys:
        return False

    for mkey in d0_mkeys:
        d0_mdesc, d1_mdesc = d0.metadata.get(mkey, 'description'), d1.metadata.get(mkey, 'description')
        d0_mval, d1_mval = d0.metadata.get(mkey, 'value'), d1.metadata.get(mkey, 'value')

        if not d0_mdesc == d1_mdesc:
            return False

        try:
            np.testing.assert_equal(d0_mval, d1_mval)
        except AssertionError:
            return False

    return True
