'''Distributed computation module

This file is a part of BdPy.
'''


__all__ = ['DistComp']


import os
import warnings
import sqlite3
from contextlib import closing


class DistComp(object):
    '''Distributed computation class'''

    def __init__(self, backend='file', comp_id=None, lockdir='tmp', db_path='./distcomp.db'):
        self.__backend = backend # 'file' or 'sqlite3'
        self.lockdir = lockdir
        self.comp_id = comp_id
        self.__db_path = db_path

        self.lockfile = self.__lockfilename(self.comp_id) if self.comp_id != None else None

        if self.__backend == 'sqlite3':
            if not os.path.isfile(self.__db_path):
                self.__init_db()

    def islocked(self, *args):
        if self.__backend == 'file' and len(args) > 0:
            raise RuntimeError('File backend does not requires computation ID.')
        if self.__backend == 'sqlite3' and len(args) != 1:
            raise RuntimeError('SQLite3 backend requires computation ID.')

        if self.__backend == 'file':
            if os.path.isfile(self.lockfile):
                return True
            else:
                return False
        elif self.__backend == 'sqlite3':
            comp_id = args[0]
            if self.__status_db(comp_id) == 'locked':
                return True
            else:
                return False
        else:
            raise ValueError('Unknown backend: %s' % self.__backend)

    def lock(self, *args):
        if self.__backend == 'file' and len(args) > 0:
            raise RuntimeError('File backend does not requires computation ID.')
        if self.__backend == 'sqlite3' and len(args) != 1:
            raise RuntimeError('SQLite3 backend requires computation ID.')

        if self.__backend == 'file':
            with open(self.lockfile, 'w'):
                pass
        elif self.__backend == 'sqlite3':
            comp_id = args[0]
            with sqlite3.connect(self.__db_path, isolation_level='EXCLUSIVE') as db:
                try:
                    db.execute('INSERT INTO computation (name, status) VALUES ("%s", "locked")' % comp_id)
                    return True
                except db.Error:
                    print('Already locked')
                    return False
        else:
            raise ValueError('Unknown backend: %s' % self.__backend)

    def unlock(self, *args):
        if self.__backend == 'file' and len(args) > 0:
            raise RuntimeError('File backend does not requires computation ID.')
        if self.__backend == 'sqlite3' and len(args) != 1:
            raise RuntimeError('SQLite3 backend requires computation ID.')

        if self.__backend == 'file':
            try:
                os.remove(self.lockfile)
            except OSError:
                warnings.warn('Failed to unlock the computation. Possibly double running.')
        elif self.__backend == 'sqlite3':
            comp_id = args[0]
            with sqlite3.connect(self.__db_path, isolation_level='EXCLUSIVE') as db:
                try:
                    db.execute('DELETE FROM computation WHERE name = "%s"' % comp_id)
                    return True
                except db.Error:
                    print('Already unlocked')
                    return False
        else:
            raise ValueError('Unknown backend: %s' % self.__backend)

    def islocked_lock(self, *args):
        if self.__backend == 'file' and len(args) > 0:
            raise RuntimeError('File backend does not requires computation ID.')

        if self.__backend == 'sqlite3':
            raise NotImplementedError()

        is_locked = os.path.isfile(self.lockfile)
        if not is_locked:
            with open(self.lockfile, 'w'):
                pass

        return is_locked

    def __lockfilename(self, comp_id):
        '''Return the lock file path'''
        return os.path.join(self.lockdir, comp_id + '.lock')

    def __init_db(self):
        with sqlite3.connect(self.__db_path, isolation_level='EXCLUSIVE') as conn:
            c = conn.cursor()
            c.execute('CREATE TABLE computation(name TEXT UNIQUE, status TEXT)')
        return None

    def __status_db(self, comp_id):
        '''Return status of `comp_id`.'''
        with sqlite3.connect(self.__db_path, isolation_level='EXCLUSIVE') as db:
            r = [row[0] for row in db.execute('SELECT STATUS FROM computation WHERE name = "%s"' % comp_id)]
            if len(r) == 0:
                st = 'not_found'
            else:
                st = r[0]
        return st
