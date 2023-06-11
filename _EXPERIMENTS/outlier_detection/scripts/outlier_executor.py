import sys, os

odm_dir = os.path.abspath(os.path.join(os.path.realpath(__file__), "../../"))
sys.path.append(odm_dir)
from ..outlier_detector import *

config_numbers = [5, 6, 7]
percent_bad = [8, 13, 24]

# config_numbers = [25, 25, 25]
# percent_bad = [24, 24, 24]

feature_types = ["downsample", "hist", "sift", "orb"]
norm_types = ["minmax", "gaussian", "max"]


def per_task_execution(idx):
    imgs = None
    feature = None
    percent = None
    norm_type = None
    dataset = None

    # configure the dataset that is used for this task iteration
    if idx % 3 == 0:
        dataset = 1
        percent = percent_bad[0]
        # config = config_numbers[0]
        # imgs = O.DataHelper.get2D(N=100, config_num=config, randomize=True)
    elif idx % 3 == 1:
        dataset = 2
        percent = percent_bad[1]
        # config = config_numbers[1]
        # imgs = O.DataHelper.get2D(N=100, config_num=config, randomize=True)
    elif idx % 3 == 2:
        dataset = 3
        percent = percent_bad[2]
        # config = config_numbers[2]
        # imgs = O.DataHelper.get2D(N=100, config_num=config, randomize=True)

    # configure the normalization type that is used for this task iteration
    if idx % 36 < 12:
        norm_type = norm_types[0]
    elif 12 <= idx % 48 < 24:
        norm_type = norm_types[1]
    elif 24 <= idx % 48 < 36:
        norm_type = norm_types[2]

        # configure the feature type that is used for this task iteration
    if idx % 4 == 0:
        feature = feature_types[0]
        # features = O.Features.get_features(imgs,
        #                                    feature_type=feature,
        #                                    norm_type=norm_type)
    elif idx % 4 == 1:
        feature = feature_types[1]
        # features = O.Features.get_features(imgs,
        #                                    feature_type=feature,
        #                                    norm_type=norm_type)
    elif idx % 4 == 2:
        feature = feature_types[2]
        # features = O.Features.get_features(imgs,
        #                                    feature_type=feature,
        #                                    norm_type=norm_type)
    elif idx % 4 == 3:
        feature = feature_types[3]
        # features = O.Features.get_features(imgs,
        #                                    feature_type=feature,
        #                                    norm_type=norm_type)

    detector_list = ["KNN", "LOF", "COF", "HBOS"]

    arguments = {
        "contamination": percent / 100,
        "verbose": False,
        "norm_type": norm_type,
        "metric": "euclidean",
        "n_jobs": 1,
        "n_bins": 2,
        "alpha": 0.2,
        "n_clusters": 2,
        "tol": 0.2,
        "beta": 3,
        "n_neighbors": 3,
        "base_estimators": detector_list,
        "cluster_estimator": "KMeans",
        "n_estimators": 4,
        "subset_size": 0.5,
        "combination": "maximum",
        "leaf_size": 20,
        "p": 4,
        "novelty": False,
        "kernel": "linear",
        "degree": 2,
        "gamma": 0.15,
        "coef0": 3,
        "nu": 0.5,
        "shrinking": True,
        "cache_size": 25,
        "max_iter": -1,
        "max_samples": 0.5,
        "bandwidth": 2.0,
        "radius": 1.2,
        "hidden_neurons": [48, 24, 24, 48],
    }

    # if feature is downsamle then exclude ['MCD', 'GMM']
    # if feature == 'downsample':
    #     exclude = ['MCD', 'GMM']

    # run the outlier detection
    # O.OutlierDetector(run_id=feature,
    #                   imgs=imgs,
    #                   features=features,
    #                   number_bad=percent,
    #                   verbose=False,
    #                   exclude=exclude,
    #                   kwargs=arguments)

    # print out the feature, the normalization type, and the dataset
    print(
        "Running data on: Feature: {}, Norm: {}, Dataset: {}".format(
            feature, norm_type, dataset
        )
    )

    OutlierDetector.generate_std_avg_all_algs(
        n_runs=5,
        norm=norm_type,
        feature=feature,
        dataset=dataset,
        display=False,
        timing=False,
        **arguments,
    )
    print(f"task number is: {idx}")
    return


if __name__ == "__main__":
    TASK_NUM = sys.argv[1]
    TASK_NUM = int(TASK_NUM)
    per_task_execution(TASK_NUM)
