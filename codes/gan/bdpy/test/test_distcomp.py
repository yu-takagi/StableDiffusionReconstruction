'''Tests for distcomp'''


from unittest import TestCase, TestLoader, TextTestRunner

import os
import shutil

from bdpy.distcomp import DistComp


class TestUtil(TestCase):
    def test_distcomp_file(self):
        lockdir = './tmp'
        comp_id = 'test-distcomp-fs'

        if os.path.exists(lockdir):
            shutil.rmtree(lockdir)
        os.mkdir(lockdir)

        # init
        distcomp = DistComp(lockdir='./tmp', comp_id=comp_id)
        self.assertTrue(os.path.isdir(lockdir))
        self.assertFalse(distcomp.islocked())

        # lock
        distcomp.lock()
        self.assertTrue(os.path.isfile(os.path.join(lockdir,
                                                    comp_id + '.lock')))
        self.assertTrue(distcomp.islocked())

        # unlock
        distcomp.unlock()
        self.assertFalse(os.path.isfile(os.path.join(lockdir,
                                                     comp_id + '.lock')))
        self.assertFalse(distcomp.islocked())

        # islocked_lock
        distcomp.islocked_lock()
        self.assertTrue(os.path.isfile(os.path.join(lockdir,
                                                    comp_id + '.lock')))
        self.assertTrue(distcomp.islocked())

        shutil.rmtree(lockdir)

    def test_distcomp_sqlite3(self):
        db_path = './tmp/distcomp.db'
        comp_id = 'test-distcomp-sqlite3-1'

        if os.path.exists(db_path):
            os.remove(db_path)

        if not os.path.exists(os.path.dirname(db_path)):
            os.mkdir(os.path.dirname(db_path))

        # init
        distcomp = DistComp(backend='sqlite3', db_path=db_path)
        self.assertTrue(os.path.isfile(db_path))
        self.assertFalse(distcomp.islocked(comp_id))

        # lock
        distcomp.lock(comp_id)
        self.assertTrue(distcomp.islocked(comp_id))

        # unlock
        distcomp.unlock(comp_id)
        self.assertFalse(distcomp.islocked(comp_id))

        # islocked_lock
        with self.assertRaises(NotImplementedError):
            distcomp.islocked_lock(comp_id)

        os.remove(db_path)


if __name__ == '__main__':
    suite = TestLoader().loadTestsFromTestCase(TestUtil)
    TextTestRunner(verbosity=2).run(suite)
