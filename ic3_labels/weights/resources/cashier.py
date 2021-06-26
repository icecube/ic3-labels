#!/usr/bin/env python
import os
import numpy as np
import pickle

from ._hash_utils import get_hash


class Cashier(object):

    def __init__(self, file_name=".cache", pickle_protocol=3, read_only=False):
        """
        Parameters
        ----------
        file_name : str, optional
            The path to the cache file.
        pickle_protocol : int, optional
            Pickle protocol to use.
        read_only : bool, optional
            If True, will only allow reading of cache file.
        """
        self._path = os.path.abspath(file_name)
        self._read_only = read_only
        self._pickle_protocol = pickle_protocol

        # load pickled cache or create new one
        if os.path.exists(self._path):
            with open(self._path, 'rb') as handle:
                self._cache = pickle.load(handle)
        else:
            self._cache = {}

    def _write_cache(self):
        """Write the cache to file. Will overwrite existing file.

        Raises
        ------
        ValueError
            If read-only is set to True.
        """
        if self._read_only:
            raise ValueError('Cache is read-only!')

        else:
            print('Saving to:', self._path)
            with open(self._path, 'wb') as handle:
                pickle.dump(
                    self._cache, handle, protocol=self._pickle_protocol)

    def get(self, key):
        """Get item from cache by key

        Parameters
        ----------
        key : str
            The key which is used to access the cache.

        Returns
        -------
        None if key does not exist, else the cached data.
        """
        if key in self._cache:
            return self._cache[key]
        else:
            return None

    def delete(self, key):
        """

        Parameters
        ----------
        key : str
            The key to delete from the cache
        """
        if key in self._cache:
            self._cache.pop(key)
            self._write_cache()

    def set(self, key, value):
        """Set new key value to the cache

        Parameters
        ----------
        key : str
            The key which is used to access the cache.
        value : object
            The values to cache.
        """
        self._cache[key] = value
        self._write_cache()


def cache(cache_file=".cache", pickle_protocol=3, read_only=True):
    """
    Cache function results into a pickled dictionary

    Parameters
    ----------
    cache_file : str, optional
        The path to the cache file.
    pickle_protocol : int, optional
        Pickle protocol to use.
    read_only : bool, optional
        If True, will only allow reading of cache file.

    Returns
    -------
    Result of decorated function
    """
    def decorator(fn):
        def wrapped(*args, **kwargs):
            objects = (
                [fn.__name__]
                + list(args) +
                [kwargs[k] for k in sorted(kwargs.keys())]
            )
            objects_tr = []
            for o in objects_tr:
                if isinstance(o, np.ndarray):
                    objects_tr.append(o.tolist())
                else:
                    objects_tr.append(o)
            md5_key = get_hash(objects_tr)
            c = Cashier(
                file_name=cache_file,
                pickle_protocol=pickle_protocol,
                read_only=read_only,
            )
            res = c.get(md5_key)
            if res is not None:
                return res
            else:
                res = fn(*args, **kwargs)
                c.set(md5_key, res)
                return res
        return wrapped
    return decorator
