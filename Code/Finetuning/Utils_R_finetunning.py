import torch  # renamed due to weird problems
from debias_transformers import *
import pandas as pd
import numpy as np
import string

from tqdm import tqdm
from gensim.models.fasttext import FastText, load_facebook_vectors, load_facebook_model
from gensim.models import KeyedVectors

#######################################
########### LATEX Functions ###########
#######################################
import time
def time_stamp():
    timestr = time.strftime("%Y%m%d-%H%M%S")
    return timestr

def create_latex_table_weat(df_biased, df_debiased):
    latex = r"""\begin{table}[htb!]
    \begin{center}

    \begin{tabular}{|c|c|c|}
    \hline
        WEAT list &  Effect size d &  Significance p \\
       \hline
    """
    # print(latex)
    for index, row in df_biased.iterrows():
        name = row['WEAT file']
        d_biased = round(row['Effect size d'], 4)
        p_biased = round(row['Significance p'], 4)
        p_biased = 1-p_biased if p_biased >0.5 else p_biased
        d_debiased = round(df_debiased.loc[index]['Effect size d'], 4)
        p_debiased = round(df_debiased.loc[index]['Significance p'], 4)
        p_debiased = 1-p_debiased if p_debiased >0.5 else p_debiased

        d_debiased = "\\textbf{"+str(d_debiased)+"}" if d_debiased < d_biased else str(d_debiased)
        p_debiased = "\\textbf{"+str(p_debiased)+"}" if p_debiased > p_biased else str(p_debiased)

        line = name + " & $" + str(d_biased) + "  \\rightarrow " + str(d_debiased) + \
        "$ & $" + str(p_biased) + " \\rightarrow " + str(p_debiased) + "$" + r""" \\
 \hline
 """
        latex += line
    caption = r""" \end{tabular}
     \caption{WEAT results, arrow indicates before to after mitigating bias}

\label{tab2}
\end{center}
\end{table} """
    # print (latex+caption)
    return latex + caption


def create_latex_table_cluster(cluster_dict):
    model_names = cluster_dict.keys()
    header =  r"""\begin{table}[htb!]
    \begin{center}

    \begin{tabular}{|c|c|c|}
    \hline 
        Model &  Original &  Debiased \\
       \hline
    """
    row = ""
    for model in model_names:
        row = model + " & " + str(cluster_dict[model][0]) + " & " + str(cluster_dict[model][1]) + r""" \\
 \hline
 """
        header += row
    caption = r""" \end{tabular}
     \caption{Cluster test results, before and after debias step}

\label{tab2}
\end{center}
\end{table} """
    return header+caption
        

def create_latex_table_downstream(downstream_dict):
    model_names = downstream_dict.keys()
    header =  r"""\begin{table}[htb!]
    \begin{center}

    \begin{tabular}{|c|c|c|}
    \hline 
        Model &  Original &  Debiased \\
       \hline
    """
    row = ""
    for model in model_names:
        row = model + " & " + str(downstream_dict[model][0]) + " & " + str(downstream_dict[model][1]) + r""" \\
 \hline
 """
        header += row
    caption = r""" \end{tabular}
     \caption{Downstream task results, before and after debias step}

\label{tab2}
\end{center}
\end{table} """
    return header+caption

def get_random_state():
    random_state = 1
    return random_state

def create_latex_table(df_biased, df_debiased):
    latex = r"""\begin{table}[htb!]
    \begin{center}

    \begin{tabular}{|c|c|c|}
    \hline
        SEAT list &  Effect size d &  Significance p \\
       \hline
    """
    # print(latex)
    for index, row in df_biased.iterrows():
        name = row['SEAT file']
        name = name.replace(
            'sent-weat_', 'SEAT-').replace('_en_nl.csv', '')
        
       
        p_biased = row['Significance p']
        p_biased = 1.0-p_biased if p_biased > 0.5 else p_biased  
        p_biased = round(p_biased,4)

        p_debiased = df_debiased.loc[index]['Significance p']
        p_debiased = 1.0-p_debiased if p_debiased > 0.5 else p_debiased
        p_debiased = round(p_debiased,4)

        d_biased = round(row['Effect size d'], 4)
        d_debiased = round(df_debiased.loc[index]['Effect size d'], 4)

        d_debiased = "\\textbf{"+str(d_debiased)+"}" if d_debiased < d_biased else str(d_debiased)
        p_debiased = "\\textbf{"+"{:.4f}".format(p_debiased)+"}" if p_debiased > p_biased else "{:.4f}".format(round(p_debiased,4))

        line = name + " & $" + str(d_biased) + "  \\rightarrow " + str(d_debiased) + "$ & $" + str(p_biased) + " \\rightarrow " + str(p_debiased) + " $" + r""" \\
 \hline
 """
        latex += line
    caption = r""" \end{tabular}
     \caption{BERTJe SEAT results, arrow indicates before to after mitigating bias}

\label{tab2}
\end{center}
\end{table} """
    # print (latex+caption)
    return latex + caption

  ##########################################################
  ########## Sentences embeddings code #####################
  ##########################################################


def encode(model, tokenizer, sentences):
    encodings = {}
    i = -1
    index = []
    for sen in tqdm(sentences):
        i += 1
        if sen in encodings.keys():
            index.append(i)
        tokenized = tokenizer.tokenize(sen)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized)
        segment_idxs = [0] * len(tokenized)
        # if len(tokenized) <5:
        #     print('error',len(tokenized))
        outputs = model(
            torch.tensor(
                [indexed_tokens]), torch.tensor(
                [segment_idxs]))
                #sequence output cls token  -Rodrigo
        # print(outputs.shape)
        # print(outputs[0])
        # enc =outputs[1]
        enc = enc[:, 0, :]  # extract the last rep of the first input
        
        # detach stops tracking possible operations? view(-1) makes it a
        # 1d array and -1 means to infer the length, numpy changes the
        # tensor to np array -R
        new_key = sen + '-i' + str(i)
        encodings[new_key] = enc.detach().view(-1).numpy()
        # print(len(encodings[sen]))
        # if len(encodings[sen])<5:
        #     print(encodings[sen])
        # index.append(i)
    # print(len(encodings))
    return encodings

    ##########################################################
    ##########  USED IN  WEAT for Preprocessing ##############
    ##########################################################


def has_punct(w):
    if any([c in string.punctuation for c in w]):
        return True
    return False


def has_digit(w):
    if any([c in '0123456789' for c in w]):
        return True
    return False


def limit_vocab(model, exclude=None, vec_len=300,limit_vocab_amount = 50000):# fasttext has 2000000 vocab, reduced for testing
    vocab_limited = []
    # limit_vocab_amount = 50000  
    for w in tqdm(model.index2entity[:limit_vocab_amount]):
        if w.lower() != w:  # check word is lower case  -R #Do we need them to be lowercase? -R
            continue
        if len(w) >= 20:  # no big words?  -R
            continue
        if has_digit(w):  # check if it has a number  -R
            continue
        if '_' in w:
            p = [has_punct(subw) for subw in w.split('_')]
            if not any(p):
                vocab_limited.append(w)
            continue
        if has_punct(w):
            continue
        vocab_limited.append(w)

    if exclude:
        vocab_limited = list(set(vocab_limited) - set(exclude))
    # just make sure unique values only
    vocab_limited = list(set(vocab_limited))
    print("size of vocabulary:", len(vocab_limited))

    wv_limited = np.zeros((len(vocab_limited), vec_len))
    for i, w in enumerate(vocab_limited):
        wv_limited[i, :] = model[w]

    return vocab_limited, wv_limited
