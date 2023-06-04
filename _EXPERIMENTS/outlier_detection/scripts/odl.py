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

OUTPUTDIR = '/raid/mpsych/ODL/CSTAR/'

CUSTOM_CONFIG = {
    # 'KNN': {
    #     'contamination': 0.07236974600054445,
    #     'leaf_size': 83,
    #     'method': 'median',
    #     'metric': 'l1',
    #     'n_jobs': 2,
    #     'n_neighbors': 44,
    #     'p': 6,
    #     'radius': 2.0,
    # },
    # 'HBOS': {
    #     'contamination': 0.07245240825856468,
    #     'alpha': 0.869355916285306,
    #     'n_bins': 50,
    #     'tol': 0.6990785947415082,
    # },
    # 'IForest': {
    #     'contamination': 0.07799920231942263,
    #     'max_features': 80,
    #     'behaviour': 'new',
    #     'bootstrap': False,
    #     'max_samples': 214,
    #     'n_estimators': 92,
    #     'n_jobs': 2,
    #     'random_state': 75,
    # },
    # 'CBLOF': {
    #     'contamination': 0.09244107318881203,
    #     'alpha': 0.5699050795721247,
    #     'beta': 2.347918956980433,
    #     'n_jobs': 3,
    #     'use_weights': True,
    # },
    'norm': 'minmax',
    'feat': 'hist',
    'accuracy_score': False,
    'return_decision_function': False,
    # 'sigma': 10,
}

NO_RUNS = 1
DATASET = 'A'
CONFIG = 'default'  # or 'best'
JOBID = 0
PRELOAD = 0  # or 1 (preload data)
FILE_NAME_ID = ''
IMGS = None
FEATURE_VECTOR = None
GT = None # ground truth

if len(sys.argv) > 1:
    NO_RUNS = sys.argv[1]
    DATASET = sys.argv[2]
    CONFIG = sys.argv[3]
    JOBID = sys.argv[4]
    PRELOAD = int(sys.argv[5])
    # check if there is an args 6 and if so make it the FILE_NAME_ID
    if len(sys.argv) > 6:
        FILE_NAME_ID = sys.argv[6]

print('STARTING ODLite..')
print('DATASET', DATASET)
print('CONFIG', CONFIG)
print('NO_RUNS', NO_RUNS)
print('JOBID', JOBID)
print('PRELOAD', PRELOAD)
print('CUSTOM_CONFIG', CUSTOM_CONFIG)

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
                                             feature_type=custom_config['feat'],
                                             norm_type=custom_config['norm'],
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
    if FILE_NAME_ID != '':
        outputfilename = DATASET + '_' + CONFIG + '_' + FN + '_' + NO_RUNS + '_' + JOBID + '_' + FILE_NAME_ID + '.pkl'
    else:
        outputfilename = DATASET + '_' + CONFIG + '_' + FN + '_' + NO_RUNS + '_' + JOBID + '.pkl'
else:
    if FILE_NAME_ID != '':
        outputfilename = DATASET + '_' + CONFIG + '_' + NO_RUNS + '_' + JOBID + '_' + FILE_NAME_ID + '.pkl'
    else:
        outputfilename = DATASET + '_' + CONFIG + '_' + NO_RUNS + '_' + JOBID + '.pkl'

# TODO: Make function to save to file to cut down on code duplication

with open(os.path.join(OUTPUTDIR, outputfilename), 'wb') as f:
    pickle.dump(runs, f)

print('All done! Stored to', outputfilename)