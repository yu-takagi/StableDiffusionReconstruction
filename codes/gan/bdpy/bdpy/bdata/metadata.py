"""
MetaData class

This file is a part of BdPy
"""


import numpy as np


class MetaData(object):
    """
    MetaData class

    'MetaData' is a list of dictionaries. Each element has three keys: 'key',
    'value', and 'description'.
    """


    def __init__(self, key=None, value=None, description=None):
        if key is None:
            key = []
        if value is None:
            value = np.ndarray((0, 0), dtype=float)
        if description is None:
            description = []

        self.__key = key
        self.__value = value
        self.__description = description

    @property
    def key(self):
        return self.__key

    @key.setter
    def key(self, x):
        self.__key = x

    @property
    def value(self):
        return self.__value

    @value.setter
    def value(self, x):
        self.__value = x

    @property
    def description(self):
        return self.__description

    @description.setter
    def description(self, x):
        self.__description = x

    def set(self, key, value, description, updater=None):
        """
        Set meta-data with `key`, `description`, and `value`

        Parameters
        ----------
        key : str
            Meta-data key
        value : array_like
            Meta-data value
        description : str
            Meta-data description
        updater : function
            Function applied to meta-data value when meta-data named `key` already exists.
            It should take two args: new and old meta-data values.
        """

        # If `value` is None, `set` does not update the value.
        is_novalue = True if value is None else False

        value = np.array(value)

        if key in self.__key:
            # Update existing metadata

            ind = [i for i, k in enumerate(self.__key) if k == key]

            if len(ind) > 1:
                raise ValueError('Multiple meta-data with the same key is not supported')

            ind = ind[0]

            self.__description[ind] = description

            # If `value` is None, `set` does not update the value.
            if is_novalue:
                return None

            if value.shape[0] > self.get_value_len():
                cols = np.empty((self.__value.shape[0], value.shape[0] - self.get_value_len()))
                cols[:] = np.nan

                self.__value = np.hstack([self.__value, cols])

            if updater is None:
                self.__value[ind, :] = value
            else:
                self.__value[ind, :] = np.array(updater(value, self.__value[ind, :]), dtype=np.float)
        else:
            # Add new metadata
            self.__key.append(key)
            self.__description.append(description)

            if value.shape[0] > self.get_value_len():
                cols = np.empty((self.__value.shape[0], value.shape[0] - self.get_value_len()))
                cols[:] = np.nan

                self.__value = np.hstack([self.__value, cols])

            self.__value = np.vstack([self.__value, value])


    def get(self, key, field):
        """
        Returns meta-data specified by `key`

        Parameters
        ----------
        key : str
            Meta-data key
        field : str
            Field name of meta-data (either 'value' or 'description')

        Returns
        -------
        array, str or None
            Meta-data value or description. If `key` was not found in
            the metadata, `None` is returned.
        """
        if key in self.__key:
            ind = self.__key.index(key)
        else:
            return None

        if field == 'value':
            return self.__value[ind, :].astype(np.float)

        if field == 'description':
            return self.__description[ind]


    def get_value_len(self):
        """Returns length of meta-data value"""
        return self.__value.shape[1]


    def keylist(self):
        """Returns a list of keys"""
        return self.__key
