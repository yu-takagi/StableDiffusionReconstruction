'''learning module'''


from abc import ABCMeta, abstractmethod

import os
import warnings
import uuid
import pickle
import copy
import yaml
import glob
from time import time, sleep
from datetime import datetime

import numpy as np

from bdpy.dataform import save_array, load_array
from bdpy.distcomp import DistComp
from bdpy.util import makedir_ifnot


#-----------------------------------------------------------------------
class BaseLearning(object):
    '''Base class for learning'''

    __metaclass__ = ABCMeta

    def __init__(self):
        self._preprocessing = []
        self._postprocessing = []

    @abstractmethod
    def run(self, *args, **kargs):
        pass

    def add_preprocessing(self, func, args=None):
        '''Add preprocessing function'''
        self._preprocessing.append({'func' : func,
                                    'args' : args})


    def add_postprocessing(self, func, args=None):
        '''Add postprocessing function'''
        self._postprocessing.append({'func' : func,
                                     'args' : args})


#-----------------------------------------------------------------------
class Classification(BaseLearning):
    '''Classification class
    Parameters
    ----------
    x_train, y_train : array_like
        Training data (features) and target labels
    x_test, y_test : array_like
        Test data (features) and target labels
    classifier
        Classifier
    verbose : {'off', 'info'}, optional
        Verbosity level

    Attributes
    ----------
    classifier_trained
        Trained classifier
    prediction
        Predicted labels
    prediction_accuracy
        Prediction accuracy
   '''


    def __init__(self, x_train=None, y_train=None, x_test=None, y_test=None,
                 classifier=None, verbose='off'):
        BaseLearning.__init__(self)

        # Parameters
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.classifier = classifier
        self.verbose = verbose

        # Results
        self.classifier_trained = None
        self.prediction = None
        self.prediction_accuracy = None


    def run(self):
        '''Run classification'''

        self.classifier_trained = copy.deepcopy(self.classifier)

        for p in self._preprocessing:
            func = p['func']
            args = p['args']

            if args == None:
                self.x_train, self.y_train, self.x_test, self.y_test \
                    = func(self.x_train, self.y_train, self.x_test, self.y_test)
            else:
                self.x_train, self.y_train, self.x_test, self.y_test \
                    = func(self.x_train, self.y_train, self.x_test, self.y_test, *args)

        self.classifier_trained.fit(self.x_train, self.y_train)
        self.prediction = self.classifier_trained.predict(self.x_test)

        self.prediction_accuracy = self.__calc_accuracy(self.prediction, self.y_test)


    def __calc_accuracy(self, ypred, ytest):
        return float(np.sum(ytest == ypred)) / len(ytest)


#-----------------------------------------------------------------------
class CrossValidation(BaseLearning):
    '''Cross-validation class

    Parameters
    ----------
    x, y : array_like
        Data (features) and target labels
    classifier :
        Classifier
    index : k-folds iterator
        Index iterator for cross-validation
    keep_classifiers : bool, optional
        If True, keep trained classifiers in each fold (default: False)
    verbose : {'off', 'info'}, optional
        Verbosity level (default: 'off')

    Attributes
    ----------
    classifier_trained : list
        Trained classifier in each fold
    prediction_accuracy : list
        Prediction accuracy in each fold
    '''

    def __init__(self, x, y, classifier=None, index=None,
                 keep_classifiers=False, verbose='off'):
        BaseLearning.__init__(self)

        # Parameters
        self.x = x
        self.y = y
        self.classifier = classifier
        self.index = index
        self.keep_classifiers = keep_classifiers
        self.verbose = verbose

        # Results
        self.classifier_trained = []
        self.prediction_accuracy = []


    def run(self):
        '''Run cross-validation

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        cls = Classification(x_train=None, y_train=None, x_test=None, y_test=None,
                             classifier=self.classifier, verbose='off')
        for p in self._preprocessing:
            func = p['func']
            args = p['args']

            if args == None:
                cls.add_preprocessing(func)
            else:
                cls.add_preprocessing(func, args=args)

        for train_index, test_index in self.index:
            cls.x_train = self.x[train_index, :]
            cls.y_train = self.y[train_index, :].flatten()
            cls.x_test  = self.x[test_index, :]
            cls.y_test  = self.y[test_index, :].flatten()

            cls.run()

            if self.keep_classifiers:
                self.classifier_trained.append(cls.classifier_trained)

            self.prediction_accuracy.append(cls.prediction_accuracy)

        if self.verbose == 'info':
            print('Prediction accuracy: %f' % np.mean(self.prediction_accuracy))


#-----------------------------------------------------------------------
class ModelTraining(object):
    '''Model training with chunking and distributed computation class.

    Parameters
    ----------
    model
       Prediction model instance
    X, Y : array_like
       Input and target data

    Attributes
    ----------
    id : str
        Training ID
    model_parameters : dict
        Parameters of the models. This will be passed to model.fit().
    X_normalize, Y_normalize : dict
        Normalization parameters for X and Y.
    X_sort : dict
        Sorting parameters for X.
    Y_sort : dict
        Sorting parameters for Y.
    dtype
        Data type (e.g., np.float32)
    chunk_axis
        The training will be divided into chunks along `chunk_axis`.
    distcomp : DistComp instance
    save_format : str ('pickle' of 'bdmodel')
    save_path : str
    verbose : int (0 or 1)
        Verbosity level.
    '''

    def __init__(self, model, X, Y):
        # Required properties
        self.model = model    # Model instance
        self.X = X            # Input, shape = (n_samples, n_features)
        self.Y = Y            # Target variables, shape = (n_samples, n_variables)

        # X and Y preprocessing parameters
        self.X_normalize = None
        self.Y_normalize = None
        self.X_sort = None
        self.Y_sort = None

        # Optional properties
        self.id = str(uuid.uuid4())
        self.model_parameters = {}     # Parameters passed to model.fit()
        self.dtype = None              # Data type
        self.chunk_axis = None         # Axis along which Y is chunked
        self.distcomp = None           # Distributed computation controller
        self.save_format = 'pickle'    # {'pickle', 'bdmodel'}
        self.save_path = './model.pkl' # Output path
        self.verbose = 1               # Verbosity level [0, 1]

        # Private members
        self.__chunking = False

    def run(self):
        '''Run training.'''

        if self.dtype is not None:
            self.X = self.X.astype(self.dtype)
            self.Y = self.Y.astype(self.dtype)

        # Chunking
        if self.chunk_axis is None:
            self.__chunking = False
        elif self.Y.ndim == 2:
            self.__chunking = False
        else:
            self.__chunking = True

        if self.__chunking:
            chunk_index = range(self.Y.shape[self.chunk_axis])
        else:
            chunk_index = [None]

        # Distributed computation setup
        if self.distcomp is None:
            dist_db_path = os.path.join(os.path.dirname(self.save_path), self.id + '.db')
            makedir_ifnot(os.path.dirname(dist_db_path))
            distcomp = DistComp(backend='sqlite3', db_path=dist_db_path)
        else:
            distcomp = self.distcomp

        # X normalization
        if not self.X_normalize is None:
            print('Normalizing X')
            self.X = (self.X - self.X_normalize['mean']) / self.X_normalize['std']
            self.X[np.isinf(self.X)] = 0

        if not self.X_sort is None:
            print('Sorting X')
            self.X = self.X[self.X_sort['index'], :]

        # Model training loop
        time_elapsed = []
        output_files_all = []

        for i, i_chunk in enumerate(chunk_index):
            loop_start_time = time()

            if self.id is None:
                training_id_chunk = 'chunk%08d' % i
            else:
                training_id_chunk = '%s-chunk%08d' % (self.id, i)

            # Output file setting
            output_files = self.__output_file(chunk=i)
            output_files_all.extend(output_files)

            # Check chunk results
            if self.__is_done(output_files):
                if self.verbose >= 1: print('%s is already done. Skipped.' % training_id_chunk)
                continue

            # Parallel computation setup
            # DistComp.lock() returns True if the computation is not locked and successfully locked.
            if not distcomp.lock(training_id_chunk):
                if self.verbose >= 1: print('%s is already running. Skipped.' % training_id_chunk)
                continue

            if self.__chunking:
                Y = np.take(self.Y, [i_chunk], axis=self.chunk_axis)
            else:
                Y = self.Y

            # Y preprocessing
            if not self.Y_normalize is None:
                print('Normalizing Y')
                if self.__chunking:
                    y_mean = np.take(self.Y_normalize['mean'], [i_chunk], axis=self.chunk_axis)
                    y_norm = np.take(self.Y_normalize['std'], [i_chunk], axis=self.chunk_axis)
                else:
                    y_mean = self.Y_normalize['mean']
                    y_norm = self.Y_normalize['std']
                Y = (Y - y_mean) / y_norm
                Y[np.isinf(Y)] = 0

            if not self.Y_sort is None:
                print('Sorting Y')
                Y = Y[self.Y_sort['index'], :]

            # Training
            if self.verbose >= 1: print('Training: %s' % training_id_chunk)
            self.model.fit(self.X, Y, **self.model_parameters)

            # Save models
            self.__save_model(output_files)

            etime = time() - loop_start_time
            time_elapsed.append(etime)
            if self.verbose >= 1: print('Elapsed time: %f' % etime)

            distcomp.unlock(training_id_chunk)

            if len(chunk_index) > 1:
                etime_ave = np.mean(time_elapsed)
                est_time_left = etime_ave * (len(chunk_index) - (i + 1))
                est_time_end = time() + est_time_left
                print('')
                print('Average computation time/chunk: %f s' % etime_ave)
                print('Estimated remaining time:       %f s' % est_time_left)
                print('Estimated computation end time: %s' % datetime.fromtimestamp(est_time_end).strftime('%Y-%m-%d %H:%M:%S'))
                print('')

        # Check outputs and add information
        if self.__is_done(output_files_all):
            if os.path.isdir(self.save_path):
                info_file = os.path.join(self.save_path, 'info.yaml')

                if os.path.exists(info_file):
                    while True:
                        with open(info_file, 'r') as f:
                            info = yaml.load(f)
                        if info is None:
                            print('Failed to load info from %s. Retrying ...' % info_file)
                            sleep(1)
                        else:
                            print('Loaded info from %s' % info_file)
                            break
                else:
                    info = {}

                if not '_status' in info:
                    info.update({'_status': {}})

                info['_status'].update({'computation_id':     self.id,
                                        'computation_status': 'done'})

                with open(info_file, 'w') as f:
                    f.write(yaml.dump(info, default_flow_style=False))

        return self.model

    def __save_model(self, output_files):
        if self.save_format == 'pickle':
            save_file = 'hoge.pkl'
            if len(output_files) != 1:
                raise RuntimeError('Invalid output file(s)')
            save_file = output_files[0]['file_path']

            makedir_ifnot(os.path.dirname(save_file))
            with open(save_file, 'wb') as f:
                pickle.dump(self.model, f, protocol=2)
            if self.verbose >= 1: print('Saved %s' % save_file)
        elif self.save_format == 'bdmodel':
            if not self.model.__class__.__name__ == 'FastL2LiR':
                raise NotImplementedError('BD model current supports only FastL2LiR models.')

            for s in output_files:
                makedir_ifnot(os.path.dirname(s['file_path']))
                save_array(s['file_path'], getattr(self.model, s['src']), key=s['dst'], dtype=self.dtype, sparse=s['sparse'])
                if self.verbose >= 1: print('Saved %s' % s['file_path'])
        else:
            raise ValueError('Unsupported output format: %s' % self.save_format)
        return None

    def __output_file(self, chunk=0):
        '''Define output files.'''
        output_files = []
        if self.save_format == 'pickle':
            # Save the model instance as pickle.
            if self.__chunking:
                save_dir = os.path.splitext(self.save_path)[0]
                output_files.append({'file_path': os.path.join(save_dir, '%08d.pkl' % chunk),
                                     'src': None,
                                     'dst': None,
                                     'sparse': False})
            else:
                output_files.append({'file_path': self.save_path,
                                     'src': None,
                                     'dst': None,
                                     'sparse': False})
        elif self.save_format == 'bdmodel':
            # Save W and b for FastL2LiR model.
            # Otherwise, save everythings.

            if self.model.__class__.__name__ == 'FastL2LiR':
                save_dir = os.path.splitext(self.save_path)[0]
                if self.__chunking:
                    save_file_W = os.path.join(save_dir, 'W', '%08d.mat' % chunk)
                    save_file_b = os.path.join(save_dir, 'b', '%08d.mat' % chunk)
                else:
                    save_file_W = os.path.join(save_dir, 'W.mat')
                    save_file_b = os.path.join(save_dir, 'b.mat')

                output_files = [
                    {'file_path': save_file_W, 'src': '_FastL2LiR__W', 'dst': 'W', 'sparse': True},
                    {'file_path': save_file_b, 'src': '_FastL2LiR__b', 'dst': 'b', 'sparse': False},
                    ]
            else:
                raise NotImplementedError('BD model current supports only FastL2LiR models.')
        else:
            raise ValueError('Unknown save format: %s' % self.save_format)
        return output_files

    def __is_done(self, output_files):
        check_outputs = [os.path.exists(out_file['file_path']) for out_file in output_files]
        return all(check_outputs)


#-----------------------------------------------------------------------
class ModelTest(object):
    '''Model test (prediction) class.'''

    def __init__(self, model, X):
        self.model = model
        self.X = X

        self.id = str(uuid.uuid4())
        self.model_format = 'pickle'
        self.model_path = None
        self.model_parameters = {}     # Parameters passed to model.predict()
        self.dtype = None
        self.chunk_axis = None
        self.verbose = 1

    def run(self):
        '''Run test.'''

        if self.dtype is not None:
            self.X = self.X.astype(self.dtype)

        if self.model_path is None:
            y_pred = self.model.predict(self.X, **self.model_parameters)
            return y_pred

        if self.model_format == 'pickle':
            if os.path.isfile(self.model_path):
                model_files = [self.model_path]
            elif os.path.isdir(self.model_path):
                model_files = sorted(glob.glob(os.path.join(self.model_path, '*.pkl')))
            else:
                raise ValueError('Invalid model path: %s' % self.model_path)
        elif self.model_format == 'bdmodel':
            if os.path.isfile(self.model_path):
                raise ValueError('BDmodel should be specified as a directory, not a file')

            # W: shape = (n_voxels, shape_features)
            if os.path.isdir(os.path.join(self.model_path, 'W')):
                W_files = sorted(glob.glob(os.path.join(self.model_path, 'W', '*.mat')))
            elif os.path.isfile(os.path.join(self.model_path, 'W.mat')):
                W_files = [os.path.join(self.model_path, 'W.mat')]
            else:
                raise RuntimeError('W not found.')

            # b: shape = (1, shape_features)
            if os.path.isdir(os.path.join(self.model_path, 'b')):
                b_files = sorted(glob.glob(os.path.join(self.model_path, 'b', '*.mat')))
            elif os.path.isfile(os.path.join(self.model_path, 'b.mat')):
                b_files = [os.path.join(self.model_path, 'b.mat')]
            else:
                raise RuntimeError('b not found.')

            model_files = [(w, b) for w, b in zip(W_files, b_files)]

        else:
            raise ValueError('Unknown model format: %s' % self.model_format)

        # Prediction loop
        y_pred_list = []
        for i, model_file in enumerate(model_files):
            print('Chunk %d' % i)
            start_time = time()

            if self.model_format == 'pickle':
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
            elif self.model_format == 'bdmodel':
                W = load_array(model_file[0], key='W')
                b = load_array(model_file[1], key='b')
                model = self.model
                model.W = W
                model.b = b
            else:
                raise ValueError('Unknown model format: %s' % self.model_format)

            y_pred = model.predict(self.X, **self.model_parameters)
            y_pred_list.append(y_pred)

            print('Elapsed time: %f s' % (time() - start_time))

        if self.chunk_axis is None:
            return y_pred_list[0]
        else:
            return np.concatenate(y_pred_list, axis=self.chunk_axis)
