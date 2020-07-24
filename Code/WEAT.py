import pandas as pd
from WEAT_TEST import WEATTests
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations, filterfalse
import numpy as np
import random
import weat_seat_common as common_weat


import Utils_R as utils_r
random.seed(utils_r.get_random_state()) 

def weat_effect_size(X, Y, A, B, embd):
    """Computes the effect size for the given list of association
     and target word pairs
       Arguments
                X, Y : List of association words
                A, B : List of target words
                embd : Dictonary of word-to-embedding for all words
       Returns
                Effect Size
    """

    Xmat = np.array([embd[w.lower()] for w in X if w.lower() in embd])
    Ymat = np.array([embd[w.lower()] for w in Y if w.lower() in embd])
    Amat = np.array([embd[w.lower()] for w in A if w.lower() in embd])
    Bmat = np.array([embd[w.lower()] for w in B if w.lower() in embd])

    XuY = list(set(X).union(Y))
    XuYmat = []
    for w in XuY:
        if w.lower() in embd:
            XuYmat.append(embd[w.lower()])
    XuYmat = np.array(XuYmat)

    d = ((np.mean(common_weat.cos_sim_W_A_B(Xmat, Amat, Bmat)) -
          np.mean(common_weat.cos_sim_W_A_B(Ymat, Amat, Bmat))) /
         np.std(common_weat.cos_sim_W_A_B(XuYmat, Amat, Bmat)))

    return d


def weat_p_value(X, Y, A, B, embd, sample):  # default for sample was 100
    """Computes the one-sided P value for the given list of association
    and target word pairs
       Arguments
                X, Y : List of association words
                A, B : List of target words
                embd : Dictonary of word-to-embedding for all words
                sample : Number of random permutations used.
       Returns
    """
    size_of_permutation = min(len(X), len(Y))
    X_Y = X + Y
    test_stats_over_permutation = []

    Xmat = np.array([embd[w.lower()] for w in X if w.lower() in embd])
    Ymat = np.array([embd[w.lower()] for w in Y if w.lower() in embd])
    Amat = np.array([embd[w.lower()] for w in A if w.lower() in embd])
    Bmat = np.array([embd[w.lower()] for w in B if w.lower() in embd])

    if not sample:
        permutations = combinations(X_Y, size_of_permutation)
    else:
        permutations = [
            common_weat.random_permutation(X_Y, size_of_permutation)
            for s in range(sample)]

    for Xi in permutations:
        Yi = filterfalse(
            lambda w: w in Xi,
            X_Y)  # get the words not in Xi
        Ximat = np.array([embd[w.lower()]
                          for w in Xi if w.lower() in embd])
        Yimat = np.array([embd[w.lower()]
                          for w in Yi if w.lower() in embd])
        test_stats_over_permutation.append(
            common_weat.test_statistic(Ximat, Yimat, Amat, Bmat))

    unperturbed = common_weat.test_statistic(Xmat, Ymat, Amat, Bmat)

    is_over = np.array(
        [o > unperturbed for o in test_stats_over_permutation])
    # +1 due to loss of accuracy*.
    # Used in another paper and makes sense to avoid having p value of
    # 0.0000
    is_over_corrected = is_over.sum() + 1
    return is_over_corrected / is_over.size
    # return is_over.sum() / is_over.size


####################################################################
#################   Compute SEAT methods.      #####################
#################                              #####################
####################################################################


dir_results = '../Rodrigo-data/Results/'


def WEAT_Test(model, model_name, verbose=False):
    """Compute the effect-size and P value"""
    WEAT_df = pd.read_csv(
        '../Rodrigo-data/WEAT/weat_lists_en_nl_corrected.csv')
    weat_results = pd.DataFrame(
        columns=[
            'Model',
            'XYAB',
            'Effect size d',
            'Significance p','WEAT file'])
    count_weat = 1
    for test in WEATTests.get_test_lists():
        #skip test 4 and 5
        count_weat = 6 if count_weat is 4 else count_weat

        XYAB = get_WEAT_lists(test, WEAT_df)
        d = weat_effect_size(*XYAB, model)
        XYAB = get_WEAT_lists(test, WEAT_df)
        #99,999 samples instead of 5k. -R 15-july-2020
        p = weat_p_value(*XYAB, embd=model, sample=99999)
        XYAB = '-'.join(test)[:10] #10 sample words
        weat_file = 'Weat-'+str(count_weat)
        weat_row = {
            'Model': model_name,
            "XYAB": XYAB,
            'Effect size d': d,
            'Significance p': p,
            'WEAT file': weat_file}
       
        weat_results = weat_results.append(weat_row, ignore_index=True)
        if verbose:
            #  print("X-Y-A-B = ",*test)
            print(model_name)
            print('WEAT d = ', d)
            print('WEAT p = ', p)
            
        count_weat+=1

    if verbose:
        print(weat_results.to_latex(index=False, column_format="htbp"))

    weat_results.to_csv(dir_results + model_name + '_03.csv')
    return weat_results
# Return X, Y, A, B


def get_WEAT_lists(column_names, WEAT_df, language='nl'):
    return (WEAT_df[WEAT_df.category == col][language].tolist()
            for col in column_names)
