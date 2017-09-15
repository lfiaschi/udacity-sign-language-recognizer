import os, sys
import numpy as np
import pandas as pd
from asl_data import AslDb
import warnings
from hmmlearn.hmm import GaussianHMM
import math
from my_model_selectors import *
import timeit
from my_recognizer import *
from asl_utils import *

asl = AslDb()  # initializes the database


# needed later convert all features to float values
for col in asl.df:
    if col != 'speaker':
        asl.df[col] = asl.df[col].astype(np.float32)


asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
asl.df['grnd-ly'] = asl.df['left-y'] - asl.df['nose-y']
asl.df['grnd-lx'] = asl.df['left-x'] - asl.df['nose-x']

# collect the features into a list groud features
features_ground = ['grnd-rx', 'grnd-ry', 'grnd-lx', 'grnd-ly']

# Normaized features
features_norm = ['norm-rx', 'norm-ry', 'norm-lx', 'norm-ly']

lookup = asl.df.groupby('speaker').transform(lambda df: (df - df.mean()) / df.std())
asl.df = asl.df.assign(**{'norm-rx': lookup['right-x'],
                          'norm-lx': lookup['left-x'],
                          'norm-ry': lookup['right-y'],
                          'norm-ly': lookup['left-y']})

# Polar features
features_polar = ['polar-rr', 'polar-rtheta', 'polar-lr', 'polar-ltheta']
asl.df['polar-lr'] = np.sqrt((asl.df['left-x'] - asl.df['nose-x']) ** 2 + (asl.df['left-y'] - asl.df['nose-y']) ** 2)
asl.df['polar-rr'] = np.sqrt((asl.df['right-x'] - asl.df['nose-x']) ** 2 + (asl.df['right-y'] - asl.df['nose-y']) ** 2)
asl.df['polar-ltheta'] = np.arctan2((asl.df['left-x'] - asl.df['nose-x']), (asl.df['left-y'] - asl.df['nose-y']))
asl.df['polar-rtheta'] = np.arctan2((asl.df['right-x'] - asl.df['nose-x']), (asl.df['right-y'] - asl.df['nose-y']))

# Delta features
features_delta = ['delta-rx', 'delta-ry', 'delta-lx', 'delta-ly']

asl.df['delta-rx'] = asl.df['right-x'].diff().fillna(0)
asl.df['delta-ry'] = asl.df['right-y'].diff().fillna(0)
asl.df['delta-lx'] = asl.df['left-x'].diff().fillna(0)
asl.df['delta-ly'] = asl.df['left-y'].diff().fillna(0)


# My features
features_custom = ['norm-polar-lr',
                   'norm-polar-rr',
                   'polar-ltheta',
                   'polar-rtheta',
                   'norm-delta-polar-lr',
                   'norm-delta-polar-rr',
                   #                   'norm-delta-polar-ltheta',
                   #                   'norm-delta-polar-rtheta'
                   ]

asl.df['delta-polar-lr'] = asl.df['polar-lr'].diff().fillna(0)
asl.df['delta-polar-rr'] = asl.df['polar-rr'].diff().fillna(0)
asl.df['delta-polar-ltheta'] = asl.df['polar-ltheta'].diff().fillna(0)
asl.df['delta-polar-rtheta'] = asl.df['polar-rtheta'].diff().fillna(0)

lookup = asl.df.groupby('speaker').transform(lambda df: (df - df.mean()) / df.std())

asl.df = asl.df.assign(**{'norm-polar-lr': lookup['polar-lr'],
                          'norm-polar-rr': lookup['polar-rr'],
                          'norm-polar-ltheta': lookup['polar-ltheta'],
                          'norm-polar-rtheta': lookup['polar-rtheta'],
                          'norm-delta-polar-lr': lookup['delta-polar-lr'],
                          'norm-delta-polar-rr': lookup['delta-polar-rr'],
                          'norm-delta-polar-ltheta': lookup['delta-polar-ltheta'],
                          'norm-delta-polar-rtheta': lookup['delta-polar-rtheta'],
                          }
                       )


def train_all_words(features, model_selector):
    training = asl.build_training(features)  # Experiment here with different feature sets defined in part 1
    sequences = training.get_all_sequences()
    Xlengths = training.get_all_Xlengths()
    model_dict = {}
    for word in training.words:
        model = model_selector(sequences, Xlengths, word,
                               n_constant=3).select()
        model_dict[word] = model
    return model_dict

import time
def evaluate_model(features, fname, selector, selector_name):

    start = time.time()
    models = train_all_words(features, selector)
    test_set = asl.build_test(features)
    probabilities, guesses = recognize(models, test_set)
    wer = score_with_wer(guesses,test_set)
    stop = time.time()

    record = { 'time': stop - start,
               'wer': wer,
               'selector': selector_name,
               'fname': fname

    }

    return record


mapping_feature_sets = {
                           'features_ground': features_ground,
                           'features_delta': features_delta,
                           'features_norm':  features_norm,
                           'features_polar': features_polar,
                           'features_custom': features_custom
                         }

mapping_selector = {    'SelectorConstant': SelectorConstant,
                       'SelectorCV': SelectorCV,
                       'SelectorBIC': SelectorBIC,
                       'SelectorDIC':  SelectorDIC,
                    }


from joblib import Parallel, delayed

jobs = []

for selector_name, selector in mapping_selector.items():
    for feature_set_name, features in mapping_feature_sets.items():

        job = delayed(evaluate_model)(features,feature_set_name, selector, selector_name)
        jobs.append(job)


result = Parallel(-1, verbose=10)(jobs)
result = pd.DataFrame(result)
result.to_csv('model_evaluation.csv',index=False)






