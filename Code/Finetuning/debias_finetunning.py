from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import json
import Utils_R_finetunning as utils_r
from tqdm import tqdm
# Compute gender direction steps:
# 1. Get lists of pair sentences. (e.g. HE is smart, SHE is smart).
# 2. Get embeddings on pair list.
# 3. Normalize embeddings.
# 4. Get component 0 of PCA(pairs_list).
# 5. return that component 0.


def norm_dict(emd_dict):
    # normalize embeddings and (if word_level) average over time steps
    for key in emd_dict.keys():  # JUST NOTMALIZE ALL EMBEDDINGS
        ebd = emd_dict[key]
        ebd /= np.linalg.norm(ebd, axis=-1, keepdims=True)
        emd_dict[key] = ebd
    return emd_dict


def get_bias_direction(model, tokenizer, from_cache,
                       filename_cache="../../Rodrigo-data/bias_direction_k10_nl_large30k",
                       amount_gen_pairs = 30000,
                       save_to_file=False, pca_components = 10,
                       save_file_name="../../Rodrigo-data/bias_direction_k10_nl_large30k"):
    if from_cache:
        return np.load(filename_cache+'.npy')
        # return np.fromfile(filename_cache, dtype=np.float32)
        
    print('Reading sentences pairs...')
    bias_pairs = compute_gender_pairs(amount_gen_pairs)
    first_col = [row[0] for row in bias_pairs]
    second_col = [row[1] for row in bias_pairs]
    len_bias_pairs = len(bias_pairs)
    bias_pairs = None
    print('Encoding sentences first...')
    first_col_enc = utils_r.encode(
                model, tokenizer, sentences=first_col)  
    first_col = None
    print('Normalizing dict first...')
    norm_first_col_enc = norm_dict(first_col_enc)      
    first_col_enc = None
    print('Encoding sentences second...')
    second_col_enc = utils_r.encode(
        model, tokenizer, sentences=second_col)
    second_col = None
    print('Normalizing dict second...')
    norm_second_col_enc = norm_dict(second_col_enc)
    second_col_enc = None
    # fair enough -Rodrigo
    assert(len_bias_pairs == len(norm_first_col_enc))
    # pairs_vals = [list(norm_first_col_enc.values()),list(norm_second_col_enc.values())]
    print('Assert passed')
    print('PCA...')
    
    gender_dir = doPCA(
        list(
            norm_first_col_enc.values()), list(
            norm_second_col_enc.values()),num_components=pca_components).components_[:pca_components]
    print(gender_dir[0].dtype)
    print('PCA done...')
    if save_to_file:  # SAVE TO FILE
        # filename = "../Rodrigo-data/bias_direction_k10_nl_large30k"
        print('Saving...')
        np.save(save_file_name+'.npy',gender_dir)
        print('Saving succesful...')
        # b = np.fromfile(filename,dtype=np.float32)
    print('DONE')
    return gender_dir


# Sentence version
def doPCA(pairs_a, pairs_b, num_components=10):
    matrix = []
    length = min(len(pairs_a), len(pairs_b))
    for i in tqdm(range(length)):
        pair_a = pairs_a[i]
        pair_b = pairs_b[i]
        center = (pair_a + pair_b) / 2
        matrix.append(pair_a - center)
        matrix.append(pair_b - center)
    matrix = np.array(matrix)
    pca = PCA(n_components=num_components, svd_solver="full")
    pca.fit(matrix)  # Produce different results each time...
    return pca


def compute_gender_pairs(amount=30000):
    # filename='nl_gender_pairs_large_alpha_90.json' #alphanumerica values, so
    # no punctuation or special char.
    filename = '../../Rodrigo-data/gender-pairs/nl_gender_pairs_large_nonalpha.json'
    with open(filename) as jsonfile:
        loaded_pairs = json.load(jsonfile)
    # actual size is 1,342,255 which would take hours to compute
    gender_pairs = loaded_pairs[:amount]
    # for p in gender_pairs:
    #     print("1",p[0])
    #     print("2",p[1])
    # print(len(gender_pairs)) #500k test
    return gender_pairs

######################################################################
#################   Methods used to debias.      #####################
################# They require a bias direction. #####################
######################################################################
    # def drop_bias(self, u, v):#Bert
    # return u - torch.ger(torch.matmul(u, v), v) / v.dot(v)


def drop(u, v):  # -Rodrigo #OTHER?
    return u - (v * u.dot(v)) / v.dot(v)

def drop_space(sen_embd, bias_subspace, k_dim = 1):
    # print('k dim:',len(bias_subspace))
    subspace = []
    for v in bias_subspace[:k_dim]: #Compute the k dim 
        sub = (v * sen_embd.dot(v)) / v.dot(v) #Project the sentence into the vector of the subspace
        subspace.append(sub)
    mean_sub = sum(subspace)    
    # print('mean_sub',mean_sub)
    return sen_embd - mean_sub

def normalize_embd(embd):
    return embd / np.linalg.norm(embd)


def get_debiased_emd(embd_dict, bias_direction, category, k_dim = 3):
    if "male" in category.lower() or "female" in category.lower():
         return embd_dict
    for key in embd_dict.keys():
        embd = embd_dict[key]
        embd = normalize_embd(embd)
        debiased = drop_space(embd, bias_direction,k_dim)
        embd = normalize_embd(debiased)
        embd_dict[key] = embd

    return embd_dict
