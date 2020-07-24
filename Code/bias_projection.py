# Bias by projection

import pandas as pd
import numpy as np
import operator


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
# create a dictionary of the bias, before and after
# he and she only, besides hardcoded, should be expanded?  -R


def compute_bias_by_projection(wv_limited, vocab_limited, vocab_full):
    """ Project the whole list of word vectors by a male and female term
    to get a projected vocabulary on male and female terms. 
    Substract each word projected on the male space by the same word projected 
    on the female space.
    Parameters:
        vocab_limited: (list[word]) vocab of model without excluded words (gender specific words).
        wv_limited: (list[i,vector]) the vectors corresponding to the vocab_limited list.
        model: current model from gensim.
    
    Returns: 
        dict: Dictionary of word :vector with the difference between the gendered projected words.

    """
    males = np.copy(wv_limited)
    males = males.dot(vocab_full['man']) #jongen #mannelijk man hij
    # Changed on midnight 26-27 may from ze
    
    females = np.copy(wv_limited)
    females = females.dot(vocab_full['vrouw']) #meisje #vrouwelijk vrouw zij
    d = {}

    for w_index, m, f in zip(vocab_limited, males, females):
        d[w_index] = m - f
    return d

# calculate the avg bias of the vocabulary (abs) before and after
# debiasing


def report_bias(gender_bias):
    """ Report bias by gender bias projection.

    """
    bias = 0.0
    for k in gender_bias:
        bias += np.abs(gender_bias[k])
    result = bias / len(gender_bias)
    print("Report bias by projection:", result)
    return result

def get_male_and_female_lists(gender_bias_projection, size=500):
    """ Get the 500 most biased for male and female. 
        Sort to get the most biased words from the gendered projected dict.
    return:
        list(male), list(female):  
    """ 
    sorted_g = sorted(
        gender_bias_projection.items(),
        key=operator.itemgetter(1))
    female = [item[0] for item in sorted_g[:size]]# The first 500 words should be the most female biased words.
    male = [item[0] for item in sorted_g[-size:]] # The last 500 words should be the most male biased words.
    return male, female

def extract_vectors(words, model):
    """Get vectors of every word from words parameter"""
    X = [model[x] for x in words]
    X = X /np.linalg.norm(X,axis=1)[:, np.newaxis]
    
    return X

#######################################
### Visualization Functions ###########
#######################################


def visualize(vectors, words, labels, ax, title,
              random_state, num_clusters=2):
    # what's tsne?  -R #I think i just transforms it to lower dim space
    # and makes it visible for the plots -R
    X_embedded = TSNE(
        n_components=2,
        random_state=random_state).fit_transform(vectors)
    if num_clusters == 2:
        for x, l in zip(X_embedded, labels):
            if l:
                ax.scatter(x[0], x[1], marker='.', c='c')  # male
            else:
                ax.scatter(
                    x[0],
                    x[1],
                    marker='x',
                    c='darkviolet')  # female
    else:
        ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels)
    ax.text(.01, .9, title, transform=ax.transAxes, fontsize=18)



# def cluster_and_visualize_single_ignore_idk(words, X, random_state, y_true, num=2, name_graph='Dutch'):
#     """ Generate 2 clusters by using KMeans.
#      Get cluster statistic based on how accurate we can separate male and female words. 

#     we do the a= 1/2k sum 1[g'i==gi] 
#     """
#     #I think here i should test the same words in biased and debiased models, it seems when i test for debiased i get diff words.
#     fig, axs = plt.subplots(1, 1, figsize=(15, 3))
#     y_pred = KMeans(
#         n_clusters=num,
#         random_state=random_state).fit_predict(X)

#     correct = [
#         1 if item1 == item2 else 0 for (
#             item1, item2) in zip(
#             y_true, y_pred)]
#     a = sum(correct) / float(len(correct))
#     print('precision', a)
#     print('Neighborhood Metric (closer to .5 is better)', max(a, 1-a))
#     print('Total samples:', len(correct))

#     visualize(X, words, y_pred, axs, name_graph, random_state)
#     fig.show()
#     # fig.savefig(filename, bbox_inches='tight')
#     return a 


def cluster_and_visualize(words, X_bef, X_aft, random_state, y_true, num=2):

    fig, axs = plt.subplots(1, 2, figsize=(15, 3))
    #changed n_init from dafault (10) to 30
    print("30 init clustering")
    y_pred_bef = KMeans(n_clusters=num, random_state=random_state,n_init=30, n_jobs=-1).fit_predict(X_bef)
    visualize(X_bef, words, y_pred_bef, axs[0], 'Original', random_state)
    correct = [1 if item1 == item2 else 0 for (item1,item2) in zip(y_true, y_pred_bef) ]
    # print( 'precision bef', sum(correct)/float(len(correct)))
    
    a_orig = sum(correct) / float(len(correct))
    print("ORIGINAL: Model Results cluster_visualize")
    print('ORIGINAL: Precision', a_orig)
    print('ORIGINAL: Neighborhood Metric (closer to .5 is better)', max(a_orig, 1-a_orig))
    # print('ORIGINAL: Total samples:', len(correct))
#######################################################################################
#######################################################################################
#######################################################################################
    y_pred_aft = KMeans(n_clusters=num, random_state=random_state).fit_predict(X_aft)
    visualize(X_aft, words, y_pred_aft, axs[1], 'Debiased', random_state)
    correct = [1 if item1 == item2 else 0 for (item1,item2) in zip(y_true, y_pred_aft) ]

    a_debias = sum(correct) / float(len(correct))
    
    print("DEBIASED: Model Results cluster_visualize")
    print('DEBIASED: Precision', a_debias)
    print('DEBIASED: Neighborhood Metric (closer to .5 is better)', max(a_debias, 1-a_debias))
    # print( 'precision aft', sum(correct)/float(len(correct)))
    fig.show()
    # fig.savefig(filename, bbox_inches='tight')
    return [a_orig, a_debias]
