'''Utilities for Pandas dataframe

This file is a part of BdPy
'''


__all__ = ['convert_dataframe', 'append_dataframe']


import pandas as pd


def convert_dataframe(lst):
    '''Convert `lst` to Pandas dataframe

    Parameters
    ----------
    lst : list of dicts

    Returns
    -------
    Pandas dataframe
    '''

    df_lst = (pd.DataFrame([item.values()], columns=item.keys()) for item in lst)
    df = pd.concat(df_lst, axis=0, ignore_index=True)
    return df


def append_dataframe(df, **kwargs):
    '''Append a row to Pandas dataframe `df`

    Parameters
    ----------
    df : Pandas dataframe
    kwargs : key-value of data to be added in `df`

    Returns
    -------
    Pandas dataframe
    '''

    df_append = pd.DataFrame({k : [kwargs[k]] for k in kwargs})
    return df.append(df_append, ignore_index=True)
