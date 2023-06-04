#!/usr/bin/env python3

#
# each run has 1 gpu
# 128 GB ram
# all on chimera12
#
#
import os, sys, pickle

omama_dir = os.path.abspath(
    os.path.join(os.path.realpath(__file__), '../../../../'))
sys.path.append(omama_dir)
import omama as O

OUTPUTDIR = '/raid/mpsych/ODL/CSTAR/OPTIMIZED/'
DATASETS = ['CSTAR']

FEAT_TYPES = ['hist', 'sift', 'orb', 'downsample']
NORM_TYPES = ['minmax', 'gaussian', 'max', 'robust', 'zscore']
# total arrangements D * N * F = 1 * 5 * 4 = 20

def per_task_execution(r, c, idx, p):
    """
    This function is used to create all possible combinations of
    datasets, normalization types, and feature types. This is
    used to create the 100 different configurations that are
    used to run the 26 PYOD algorithms on the 5 datasets.
    """
    feature = None
    norm_type = None
    dataset = DATASETS[0]

    # if idx % 5 == 0:
    #     dataset = DATASETS[0]
    # elif idx % 5 == 1:
    #     dataset = DATASETS[1]
    # elif idx % 5 == 2:
    #     dataset = DATASETS[2]
    # elif idx % 5 == 3:
    #     dataset = DATASETS[3]
    # elif idx % 5 == 4:
    #     dataset = DATASETS[4]

    # if idx % 25 <= 5:
    #     norm_type = NORM_TYPES[0]
    # elif 5 < idx % 25 <= 10:
    #     norm_type = NORM_TYPES[1]
    # elif 10 < idx % 25 <= 15:
    #     norm_type = NORM_TYPES[2]
    # elif 15 < idx % 25 <= 20:
    #     norm_type = NORM_TYPES[3]
    # elif 20 < idx % 25 <= 25:
    #     norm_type = NORM_TYPES[4]

    if idx % 5 == 0:
        norm_type = NORM_TYPES[0]
    elif idx % 5 == 1:
        norm_type = NORM_TYPES[1]
    elif idx % 5 == 2:
        norm_type = NORM_TYPES[2]
    elif idx % 5 == 3:
        norm_type = NORM_TYPES[3]
    elif idx % 5 == 4:
        norm_type = NORM_TYPES[4]

    if idx % 20 <= 5:
        feature = FEAT_TYPES[0]
    elif 5 < idx % 20 <= 10:
        feature = FEAT_TYPES[1]
    elif 10 < idx % 20 <= 15:
        feature = FEAT_TYPES[2]
    elif 15 < idx % 20 <= 20:
        feature = FEAT_TYPES[3]

    #
    # if idx % 100 <= 25:
    #     feature = FEAT_TYPES[0]
    # elif 25 < idx % 100 <= 50:
    #     feature = FEAT_TYPES[1]
    # elif 50 < idx % 100 <= 75:
    #     feature = FEAT_TYPES[2]
    # elif 75 < idx % 100 <= 100:
    #     feature = FEAT_TYPES[3]

# def per_task_execution(r, c, idx, p):
#     """
#     This function is used to create all possible combinations of
#     datasets, with one normalization type, and one feature type.
#     """
#     dataset = None
#
#     if idx % 5 == 0:
#         dataset = DATASETS[0]
#     elif idx % 5 == 1:
#         dataset = DATASETS[1]
#     elif idx % 5 == 2:
#         dataset = DATASETS[2]
#     elif idx % 5 == 3:
#         dataset = DATASETS[3]
#     elif idx % 5 == 4:
#         dataset = DATASETS[4]
#
#     norm_type = NORM_TYPES[4]
#     feature = FEAT_TYPES[2]



    # print out the configuration for this task iteration
    print('On Iteration', idx)
    print('Dataset', dataset)
    print('norm_type', norm_type)
    print('feature', feature)

    CUSTOM_CONFIG = {
        'norm': norm_type,
        'feat': feature,
        'accuracy_score': False,
        'return_decision_function': False,
        'sigma': 20,
        'batch_size': 32,
        'capacity': 0.06131767922552495,
        'contamination': 0.015414562231089293,
        'decoder_neurons':  [16, 32, 64],
        'dropout_rate': 0.42865805527000883,
        'encoder_neurons': [64, 32, 16],
        'epochs': 75,
        'gamma': 0.04668196456317853,
        'hidden_activation': 'tanh',
        'l2_regularizer': 0.4532703965079503,
        'latent_dim': 16,
        'optimizer': 'rmsprop',
        'output_activation': 'sigmoid',
        'preprocessing': False,
        'random_state': 2019,
        'validation_size': 0.16440964344699366
    }

    NO_RUNS = r
    DATASET = dataset
    CONFIG = c
    JOBID = str(idx)
    PRELOAD = p
    IMGS = None
    FEATURE_VECTOR = None
    GT = None

    print('STARTING ODLite..')
    print('DATASET', DATASET)
    print('CONFIG', CONFIG)
    print('NO_RUNS', NO_RUNS)
    print('JOBID', JOBID)
    print('PRELOAD', PRELOAD)

    default_config = False
    custom_config = None
    if CONFIG == 'default':
        default_config = True
    if CONFIG == 'custom':
        default_config = None
        custom_config = CUSTOM_CONFIG

    odl = O.OutlierDetectorLite()

    runs = {}

    if PRELOAD == 1:
        print('Preloading data')

        IMGS = odl.load_data(DATASET)
        # print(IMGS)
        FEATURE_VECTOR = O.Features.get_features(IMGS,
                                                 feature_type=custom_config[
                                                     'feat'],
                                                 norm_type=custom_config[
                                                     'norm'],
                                                 **custom_config
                                                 )
        # print(FEATURE_VECTOR)
        GT = odl.load_ground_truth(DATASET)
        # print(GT)
        print('Done preloading data')
    else:
        print('Not preloading data')

    for algo in odl.ALGORITHMS:

        print('Running', algo)

        runs[algo] = []

        for run in range(int(NO_RUNS)):
            results = odl.run(DATASET=DATASET,
                              ALGORITHM=algo,
                              default_config=default_config,
                              custom_config=custom_config,
                              imgs=IMGS,
                              feature_vector=FEATURE_VECTOR,
                              groundtruth=GT)

            # print(algo, results['evaluation']['jaccard_score'])

            runs[algo].append(results)

        print('=' * 80)

    if CONFIG == 'custom':
        FN = custom_config['feat'] + '_' + custom_config['norm']
        outputfilename = DATASET + '_' + CONFIG + '_' + FN + '_' + NO_RUNS + '_' + JOBID + '.pkl'
    else:
        outputfilename = DATASET + '_' + CONFIG + '_' + NO_RUNS + '_' + JOBID + '.pkl'

    # makesure the OUTPUTDIR exists and if not create it and any missing parent directories
    os.makedirs(OUTPUTDIR, exist_ok=True)

    with open(os.path.join(OUTPUTDIR, outputfilename), 'wb') as f:
        pickle.dump(runs, f)


if __name__ == '__main__':
    num_runs = sys.argv[1]
    config = sys.argv[2]
    task = int(sys.argv[3])
    preload = int(sys.argv[4])
    task = int(task)
    per_task_execution(num_runs, config, task, preload)
