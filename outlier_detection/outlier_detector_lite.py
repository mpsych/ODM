import glob
import os
import pickle

import sklearn
import warnings

import outlier_detection as O
import numpy as np
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
THRESHOLD = 0.0001


class OutlierDetectorLite:

    def __init__(self, data_path='/tmp/odm/'):
        """ Initializes the class
        """
        self.data_path = data_path
        self.ALGORITHMS = [
            'VAE',
        ]
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path, exist_ok=True)

    def load_data(self, DATASET):
        """ Loads the dataset
        Parameters
        ----------
        DATASET : str
            The name of the dataset to load
        Returns
        -------
        imgs : np.ndarray
            The dataset
        """
        with open(os.path.join(self.data_path, 'dataset' + DATASET + '.pkl'),
                  'rb') as f:
            imgs = pickle.load(f)
        return imgs

    def load_configs(self, DATASET):
        """ Loads the dataset
        Parameters
        ----------
        DATASET : str
            The name of the dataset to load
        Returns
        -------
        configs : dict
            The dataset
        """
        with open(os.path.join(self.data_path,
                               'dataset' + DATASET + '_configs.pkl'),
                  'rb') as f:
            configs = pickle.load(f)
        return configs

    def load_ground_truth(self, DATASET):
        """ Loads the ground truth
        Parameters
        ----------
        DATASET : str
            The name of the dataset to load
        Returns
        -------
        ground_truth : np.ndarray
            The ground truth
        """
        with open(os.path.join(self.data_path,
                               'dataset' + DATASET + '_labels.pkl'),
                  'rb') as f:
            ground_truth = pickle.load(f)
        return ground_truth

    @staticmethod
    def load_results(resultsfile):
        """ Loads the results
        Parameters
        ----------
        resultsfile : str
            The path to the results file
        Returns
        -------
        results : dict
            The results
        """
        with open(resultsfile, 'rb') as f:
            results = pickle.load(f)

        return results

    def display_best_results(self, resultsfile):
        """ Displays the best results
        Parameters
        ----------
        results : dict
            The results
        Returns
        -------
        None
        """
        all_evals = self.load_results(resultsfile)
        best_tp = 0
        best_tp_alg = ''
        for alg in all_evals.keys():
            tp = all_evals[alg][0]['evaluation']['tp']
            print(alg, tp)
            if best_tp < tp:
                best_tp = tp
                best_tp_alg = alg

    def run(self,
            DATASET,
            ALGORITHM,
            imgs=None,
            feature_vector=None,
            groundtruth=None,
            default_config=False,
            custom_config=None):
        """ Runs the outlier detection algorithm on the dataset
    Parameters
    ----------
    DATASET : str
      The name of the dataset to run the algorithm on
    ALGORITHM : str
      The name of the algorithm to run
    imgs : np.ndarray, None
        A dataset to use instead of the default one
    feature_vector : np.ndarray, None
        A feature vector to use instead of the default one
    groundtruth : np.ndarray, None
        A ground truth to use instead of the default one
    default_config : bool
      Whether to use the default configuration or not
    custom_config : dict, None
        A custom configuration to use
    Returns
    -------
    results : dict
      The results of the algorithm
    """

        DATASETS = {'A': 0.08,
                    'B': 0.13,
                    'C': 0.24,
                    'D': 0.24,
                    'ASTAR': 0.063,
                    'BSTAR': 0.050,
                    'CSTAR': 0.015
                    }

        CONTAMINATION = DATASETS[DATASET]

        # load data
        if imgs is None:
            imgs = self.load_data(DATASET)

        print('Loaded images.')

        if custom_config is None:
            # setup algorithm w/ the best config w/ the best feat w/ best norm
            configs = self.load_configs(DATASET)

            CONFIG = configs[ALGORITHM]['config']

            if default_config:
                CONFIG = {}  # clear the config and use default
                CONFIG = {'contamination': CONTAMINATION}

            NORM = configs[ALGORITHM]['norm']
            FEAT = configs[ALGORITHM]['feat']

            print('Loaded config, norm, and feats.')
        # elseif kwargs is not empty
        elif custom_config is not None:
            CONFIG = custom_config
            NORM = CONFIG['norm']
            FEAT = CONFIG['feat']

            print('Loaded custom config, norm, and feats.')
        else:
            raise ValueError('No config provided!')

        CONFIG['verbose'] = 0
        CONFIG['return_decision_function'] = True
        CONFIG['accuracy_score'] = False
        print('config: ', CONFIG)

        print("FEAT: ", FEAT)
        print("NORM: ", NORM)

        if feature_vector is None:
            feature_vector = \
                O.Features.get_features(imgs, FEAT, NORM, **CONFIG)
        else:
            print('Using provided feature vector.')

        print('Calculated features!')

        scores, labels, decision_function = O.OutlierDetector.detect_outliers(
            features=feature_vector,
            imgs=imgs,
            pyod_algorithm=ALGORITHM,
            display=False,
            number_bad=self.__number_bad(DATASET),
            **CONFIG)

        print('Trained!')

        #
        # EVALUATE
        #

        # load groundtruth
        if groundtruth is None:
            groundtruth = self.load_ground_truth(DATASET)
        else:
            print('Using provided ground truth.')

        if labels is not None:
            if len(labels) == len(groundtruth):
                evaluation = self.evaluate(groundtruth, labels)
        else:
            evaluation = {
                'groundtruth_indices': np.where(np.array(groundtruth) > 0),
                'pred_indices': np.where(np.array(scores) > 0),
                'roc_auc': 0,
                'f1_score': 0,
                'acc_score': 0,
                'jaccard_score': 0,
                'precision_score': 0,
                'average_precision': 0,
                'recall_score': 0,
                'hamming_loss': 0,
                'log_loss': 0,
                'tn': 0,
                'fp': 0,
                'fn': 0,
                'tp': 0,
            }

        results = {
            'algorithm': ALGORITHM,
            'norm': NORM,
            'feat': FEAT,
            'dataset': DATASET,
            'scores': scores,
            'labels': labels,
            # 'decision_function': decision_function,
            'groundtruth': groundtruth,
            'evaluation': evaluation
        }

        return results

    @staticmethod
    def evaluate(groundtruth, pred):
        """ Evaluates the results of the outlier detection algorithm
    Parameters
    ----------
    groundtruth : list
      The groundtruth labels
    pred : list
      The predicted labels
    Returns
    -------
    evaluation : dict
      The evaluation metrics
    """

        cm = sklearn.metrics.confusion_matrix(groundtruth, pred)

        scores = {
            'groundtruth_indices': np.where(np.array(groundtruth) > 0),
            'pred_indices': np.where(np.array(pred) > 0),
            'roc_auc': sklearn.metrics.roc_auc_score(groundtruth, pred),
            'f1_score': sklearn.metrics.f1_score(groundtruth, pred),
            'acc_score': sklearn.metrics.accuracy_score(groundtruth, pred),
            'jaccard_score': sklearn.metrics.jaccard_score(groundtruth, pred),
            'precision_score': sklearn.metrics.precision_score(groundtruth,
                                                               pred),
            'average_precision': sklearn.metrics.average_precision_score(
                groundtruth, pred),
            'recall_score': sklearn.metrics.recall_score(groundtruth, pred),
            'hamming_loss': sklearn.metrics.hamming_loss(groundtruth, pred),
            'log_loss': sklearn.metrics.log_loss(groundtruth, pred),
            'tn': cm[0, 0],
            'fp': cm[0, 1],
            'fn': cm[1, 0],
            'tp': cm[1, 1],
        }

        return scores

