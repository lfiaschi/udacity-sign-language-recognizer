import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences
from collections import defaultdict
import pandas as pd



class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """
    Select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_model = None
        best_BIC = np.inf

        for n_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(n_states)
                logL = model.score(self.X, self.lengths)

                # The number of free parameters of hmm
                # https://discussions.udacity.com/t/number-of-parameters-bic-calculation/233235/11
                initial_state_probs = n_states - 1
                transition_probs = n_states*(n_states - 1)
                emission_probs = 2 * n_states * len(model.means_)

                n_free_params = initial_state_probs + transition_probs + emission_probs

                BIC = -2 * logL + n_free_params * np.log(self.X.shape[0])

                if self.verbose:
                    print('n_states={} BIC={} best_BIC={}'.format(n_states,BIC,best_BIC))

                if BIC < best_BIC:
                    best_model = model
                    best_BIC = BIC

            except Exception as exc:

                if self.verbose:
                    print('Model could not be learned {} because {}'.format(n_states, exc))

        return best_model


class SelectorDIC(ModelSelector):
    """
    Select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    """

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        all_dics = list()
        best_model = None

        for n_states in range(self.min_n_components, self.max_n_components + 1):

            try:
                model = self.base_model(n_states)
                logL_original = model.score(self.X, self.lengths)

                other_logLs = list()

                for word in self.words:
                    if word != self.this_word:
                        Xother, length_other = self.hwords[word]
                        other_logLs.append(model.score(Xother, length_other))

                DIC = logL_original - np.mean(other_logLs)
                if self.verbose:
                    print('n_states={}, DIC={}'.format(n_states,DIC))

                all_dics.append((DIC,model))

            except Exception as exc:
                if self.verbose:
                    print('Model could not be learned {} because {}'.format(n_states, exc))

        if all_dics:

            best_model = max(all_dics, key= lambda x: x[0])[1]

        return best_model



class SelectorCV(ModelSelector):
    """
    Select best model based on average log Likelihood of cross-validation folds
    """

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_num_hidden_states = self.min_n_components
        best_log_L = -np.inf

        for n_states in range(self.min_n_components, self.max_n_components+1):

            try:
                split_method = KFold(n_splits=min(len(self.sequences),3), random_state=self.random_state)

                row_train = list()
                row_test = list()
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):

                    X_train, lengths_train  = combine_sequences(cv_train_idx, self.sequences)
                    X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)

                    # use only training samples
                    model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=1000,
                                            random_state=self.random_state, verbose=False)\
                        .fit(X_train, lengths_train)

                    logL_train = model.score(X_train, lengths_train)
                    logL_test = model.score(X_test, lengths_test)

                    # print('n_states={}, train_logL={}, test_logL={}'.format(n_states, logL_train, logL_test))

                    row_train.append(logL_train)
                    row_test.append(logL_test)

                mlogL_train = np.mean(row_train)
                mlogL_test = np.mean(row_test)
                slogL_train= np.std(row_train)
                slogL_test = np.std(row_test)

                if self.verbose:
                    print('n_states=%d, train_logL=%.2f+=%.2f, test_logL=%.2f+=%.2f'%(n_states,mlogL_train,slogL_train,
                                                                                  mlogL_test, slogL_test))

                if mlogL_test >= best_log_L:
                    best_num_hidden_states = n_states
                    best_log_L = mlogL_test
            except Exception as exc:

                if self.verbose:
                    print('Model could not be learned {} because {}'.format(n_states,exc ))


        # finally retrain on everything to output the final model when the best number of hidden states has been
        # selected
        return self.base_model(best_num_hidden_states)









