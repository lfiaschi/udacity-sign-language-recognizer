import warnings
from asl_data import SinglesData
import random

def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Likelihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    n_words = len(test_set.get_all_Xlengths())

    for i  in range(n_words):
        X, lengths = test_set.get_item_Xlengths(i)

        scoring = dict()
        for word, model in models.items():
            try:
                scoring[word] = model.score(X,lengths)
            # some combinations cannot be scored
            except:
                pass

        probabilities.append(scoring)

        if scoring:
            best_guess = max(scoring.items(), key=lambda x: x[1])[0]

        else: #in case no correct scoring could be produced just use a random word

            best_guess = word
        guesses.append(best_guess)

    return probabilities, guesses
