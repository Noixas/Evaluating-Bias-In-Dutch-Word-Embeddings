# Bias by neighbors

import scipy.stats
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import os

import json 

# def extract_professions():
#     professions = []
#     with codecs.open('../data/lists/professions.json', 'r', 'utf-8') as f:
#         professions_data = json.load(f)
#     for item in professions_data:
#         professions.append(item[0].strip())
#     return professions


# professions = extract_professions()



# top k similar words -R
def topK(w, vocab_limited, wv_limited, model, k=10):

    # extract the word vector for word w
    vec = model[w]
    # compute similarity of w with all words in the vocabulary
    sim = wv_limited.dot(vec)
    # sort similarities by descending order
    sort_sim = (sim.argsort())[::-1]
    best = sort_sim[:(k + 1)]  # choose topK
    return [vocab_limited[i] for i in best if w != vocab_limited[i]]

# get tuples of biases and counts of masculine/feminine NN for each word (for bias-by-neighbors)
# takes (10to15)+ min approx in g4dn.2xlarge  -R


def compute_bias_by_neighbors(limit_vocab, wv_limited, gender_bias_nl,
                              model, name, neighbours_num=1000,
                              from_cache=True, to_cache=True):
    """ Compute the topk most biased words per word.
    return
        tuple(word,  vector_biased,  vector_debiased,    male_biased_count,    female_biased_count)
    """
    tuples = []
    if from_cache and os.path.isfile('../Rodrigo-data/Cached_Files/bias_by_neighbours_' +
                                     name + '_jongen.csv'):
        tuples = get_bias_by_neighbors_df_from_cache(name)
    else:
        for w in tqdm(limit_vocab):
            top = topK(
                w,
                limit_vocab,
                wv_limited,
                model,
                k=neighbours_num +
                5)
            m, f = 0, 0
            for w2 in top:
                if gender_bias_nl[w2] > 0:
                    m += 1
                else:
                    f += 1
            tuples.append(
                (w, gender_bias_nl[w], gender_bias_nl[w], m, f))  # todo: not the right output, should return debiased vector too.
        if to_cache:
            save_bias_by_neighbors(tuples, name)
    return tuples

# Cache related methods


def save_bias_by_neighbors(bias_by_neighbors, name):
    bias_by_neighbors_df = pd.DataFrame(bias_by_neighbors)
    bias_by_neighbors_df.sort_values(inplace=True, by=[2])
    filename = '../Rodrigo-data/Cached_Files/bias_by_neighbours_' + \
        name + '_jongen.csv'
    bias_by_neighbors_df.to_csv(filename)


def get_bias_by_neighbors_df_from_cache(name):
    tuples = []
    filename = '../Rodrigo-data/Cached_Files/bias_by_neighbours_' + \
        name + '_jongen.csv'
    df_tuples = pd.read_csv(filename, index_col=0)
    for index, rows in df_tuples.iterrows():  # TODO:?
        # TODO: What are this hardcoded numbers?
        row_tuple = (rows[0], rows[1], rows[2], rows[3])
        tuples.append(row_tuple)
    return tuples


########################
# EXTRA CLUSTERING


# def show_plots(tuples_bef_prof, tuples_aft_prof):
    
#     fig, axs = plt.subplots(2,1, figsize=(8,8))
    
#     for i,(tuples, title) in enumerate(zip([tuples_bef_prof, tuples_aft_prof], ['Original', 'Debiased'])):
#         X = []
#         Y = []
#         for t in tuples:
#             print (tuples)
#             X.append(t[1])
#             Y.append(t[3])
#         # print(i)
#         axs[i].scatter(X,Y)
#         axs[i].set_ylim(0,100)
#         for t in tuples: #hardcoded  -R
#             if t[0] in ['nanny', 'dancer', 'housekeeper', 'receptionist', 'nurse',\
#                    'magician', 'musician', 'warden', 'archaeologist', 'comic', 'dentist', \
#                     'inventor', 'colonel', 'farmer', 'skipper', 'commander', 'coach']:
#                 axs[i].annotate(t[0], xy=(t[1], t[3]), xytext=(t[1], t[3]), textcoords="data", fontsize=12) 
#         axs[i].text(.03, .85, title, transform=axs[i].transAxes, fontsize=20)
    
    
#     fig.show()




# fix this
def show_plots(tuples_ext):
    fig, axs = plt.subplots(1, 1, figsize=(15, 3))
    X, Y = [], []
    title = 'test'
    print(len(tuples_ext))
    tuples_ext = tuples_ext[:100]
    cdt = 0
    for t in tuples_ext:
        # print(t)
        X.append(t[1])
        Y.append(t[3])
    #    axs[i].set_ylim(0,100)
    # TODO: Fix list
    # TODO get the firts 50 and last 100 since we sort the bias now
        # print(t[0])
        # if t[0] in ['nanny', 'dancer', 'housekeeper', 'receptionist', 'nurse',\
        #             'magician', 'musician', 'warden', 'archaeologist', 'comic', 'dentist', \
        #             'inventor', 'colonel', 'farmer', 'skipper', 'commander', 'coach']:
        if cdt < 30:
            axs.annotate(
                t[0], xy=(
                    t[1], t[3]), xytext=(
                    t[1], t[3]), textcoords="data", fontsize=12)
        cdt += 1
    axs.text(.03, .85, title, transform=axs.transAxes, fontsize=20)

    axs.scatter(X, Y)
    # axs.set_ylim(0,100)

    fig.show()


# compute correlation between bias-by-projection and bias-by-neighbors
def pearson(a, b):
    return scipy.stats.pearsonr(a, b)


def compute_corr(tuples, i1, i2):
    a, b = [], []
    for t in tuples:
        a.append(t[i1])
        b.append(t[i2])
    assert(len(a) == len(b))
    print(pearson(a, b))
