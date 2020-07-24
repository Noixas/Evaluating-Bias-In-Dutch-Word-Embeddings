
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import json

from gensim.models.fasttext import FastText, load_facebook_vectors, load_facebook_model
from gensim.models import KeyedVectors

# import Utils_R as utils_r
from tqdm import tqdm
"""
Code modified from 
Hard-debias embedding

Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings
Tolga Bolukbasi, Kai-Wei Chang, James Zou, Venkatesh Saligrama, and Adam Kalai
2016
"""

def debias(model, gender_specific_words, definitional, equalize):
    gender_direction = doPCA(definitional, model).components_[0]
    # gender_direction = sum(gender_direction)
    print('Gender direction ready')
    #Todo cache this variable
    specific_set = set(gender_specific_words)
    words, index, vectors = prepare_model(model)
    print('words length', len(words))
    for i, w in enumerate(tqdm(words)):        
        if w not in specific_set: #Debias word that has nothing to do with gender
            vectors[i] = drop(vectors[i], gender_direction)

    # model[w] =  embd / np.linalg.norm(embd)
    vectors = normalize(vectors) #In previous step
    print('Debiased embedding')
    print('Equalizing step')
    candidates = {x for e1, e2 in equalize for x in [(e1.lower(), e2.lower())]}
    print(candidates)
    for (a, b) in tqdm(candidates):
        if (a in words and b in words):
            y = drop((get_vect(vectors, index, a) + get_vect(vectors, index, b)) / 2, gender_direction)
            z = np.sqrt(1 - np.linalg.norm(y)**2)
            if (get_vect(vectors, index, a) - get_vect(vectors, index, b)).dot(gender_direction) < 0:
                z = -z
            vectors[index[a]] = z * gender_direction + y
            vectors[index[b]] = -z * gender_direction + y
    print('Normalizing')
    vectors = normalize(vectors)
    new_dict = {}
    for w in words:
        new_dict[w] = vectors[index[w]]
    # return new_dict

    for word in tqdm(words):
        model.syn0[model.vocab[word].index]= vectors[index[word]]
    return model, new_dict

def get_vect(vectors, index, word):
    return vectors[index[word]]
def normalize(vectors):
    vectors /= np.linalg.norm(vectors, axis=1)[:, np.newaxis]
    return vectors

def prepare_model(model):
    words = sorted([w for w in model.vocab], key=lambda w: model.vocab[w].index)
    vectors = [model[w] for w in words]
    vectors = np.array(vectors, dtype='float32')
    index = {w: i for i, w in enumerate(words)}
    # self.n, self.d = self.vecs.shape
    assert len(words) == len(index)
    # print(self.n, "words of dimension", self.d, ":", ", ".join(self.words[:4] + ["..."] + self.words[-4:]))
    return words, index, vectors


def doPCA(pairs, embedding, num_components = 10):
    matrix = []
    print('PCA...')
    for a, b in tqdm(pairs):
        a = a.lower()
        b = b.lower()
        center = (embedding[a] + embedding[b])/2
        matrix.append(embedding[a] - center)
        matrix.append(embedding[b] - center)
    matrix = np.array(matrix)
    pca = PCA(n_components = num_components)
    pca.fit(matrix)
    # bar(range(num_components), pca.explained_variance_ratio_)
    return pca
    
def drop(u, v):
    return u - v * u.dot(v) / v.dot(v)

def drop_space(sen_embd, bias_subspace, k_dim = 1):
    # print('k dim:',len(bias_subspace))
    subspace = []
    for v in bias_subspace[:k_dim]: #Compute the k dim 
        sub = (v * sen_embd.dot(v)) / v.dot(v) #Project the sentence into the vector of the subspace
        subspace.append(sub)
    mean_sub = sum(subspace)    
    # print('mean_sub',mean_sub)
    return sen_embd - mean_sub
##############################################################################
##############################################################################
###################   LOAD LISTS FOR DEBIASING  ##############################
##############################################################################
##############################################################################
def load_gender_specific_words():        
    male = pd.read_csv('../Rodrigo-data/Clustering/male_words_en_nl_corrected.csv', index_col=0)
    female = pd.read_csv('../Rodrigo-data/Clustering/female_words_en_nl_corrected.csv', index_col=0)
    gender_specific = male.append(female).nl.tolist() #return dutch column of merged files
    return gender_specific

def load_def_and_equ_words():
        
    definitional_filename = '../Rodrigo-data/WEAT/NL_definitional_pairs.json'
    with open(definitional_filename, "r") as f:
        defs = json.load(f)
    # print("definitional", defs)

    equalize_pairs_filename = '../Rodrigo-data/WEAT/NL_equalize_pairs.json'
    with open(equalize_pairs_filename, "r") as f:
        equalize_pairs = json.load(f)
    
    return defs, equalize_pairs
