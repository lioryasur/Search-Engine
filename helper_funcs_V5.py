from contextlib import closing
import itertools
from itertools import chain
import pandas as pd
import numpy as np
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from time import time
import tqdm
import math
from nltk.metrics.distance import jaccard_distance
from nltk.util import ngrams
from index_module import create_title_dict
from nltk.stem.porter import *
stemmer = PorterStemmer()

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links", 
                    "may", "first", "see", "history", "people", "one", "two", 
                    "part", "thumb", "including", "second", "following", 
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
BLOCK_SIZE = 1999998
TUPLE_SIZE = 6       
TF_MASK = 2 ** 16 - 1 # Masking the 16 low bits of an integer


global title_dict
title_dict = create_title_dict()

def tokenize(text):
    list_of_tokens =  [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() if token.group() not in all_stopwords]
    return list_of_tokens


class MultiFileReader:
    """ Sequential binary reader of multiple files of up to BLOCK_SIZE each. """

    def __init__(self):
        self._open_files = {}
        
    def read(self, locs, n_bytes, folder):
        b = []
        for f_name, offset in locs:
            if f_name not in self._open_files:
                self._open_files[f_name] = open(f'{folder}/{f_name}', 'rb')
            f = self._open_files[f_name]
            f.seek(offset)
            n_read = np.min((n_bytes, BLOCK_SIZE - offset))
            b.append(f.read(n_read))
            n_bytes -= n_read
        return b''.join(b)

    def close(self):
        for f in self._open_files.values():
            f.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False





def read_posting_list(inverted, w):
    with closing(MultiFileReader()) as reader:
        locs = inverted.posting_locs[w]
        b = reader.read(locs, inverted.df[w] * TUPLE_SIZE, inverted.folder)
        posting_list = []
        for i in range(inverted.df[w]):
            doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
            tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
            posting_list.append((doc_id, tf))
    return posting_list


def read_posting_list_tfidf(inverted, w):
    idf = np.log10(len(inverted.DL)/np.array(inverted.df[w]))
    with closing(MultiFileReader()) as reader:
        locs = inverted.posting_locs[w]
        b = reader.read(locs, inverted.df[w] * TUPLE_SIZE, inverted.folder)
        posting_list = {}
        doc_ids = []
        for i in range(inverted.df[w]):
            doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
            tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')/inverted.DL[doc_id]
            posting_list[(doc_id, w)]= tf*idf/inverted.vec_norm[doc_id]
            doc_ids.append(doc_id)
    return posting_list, doc_ids


def generate_document_tfidf_matrix(unique_terms,index):
    res_dict = {}
    doc_ids = set([])
    for term in unique_terms:
        dic, ids = read_posting_list_tfidf(index, term)
        res_dict.update(dic)
        doc_ids = doc_ids.union(ids)
    return (res_dict, doc_ids)


def generate_query_tfidf_vector(query_to_search,index):
    """ 
    Generate a vector representing the query. Each entry within this vector represents a tfidf score.
    The terms representing the query will be the unique terms in the index.

    We will use tfidf on the query as well. 
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the query.    

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.'). 
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.    
    
    Returns:
    -----------
    vectorized query with tfidf scores
    """
    total_query_size = len(set(query_to_search))
    Q = np.zeros((total_query_size))
    term_vector = np.array(sorted(list(set(query_to_search))))
    counter = Counter(query_to_search)
    tf = np.array(list(counter.values()))/len(query_to_search)
    df = np.array([index.df[term] for term in term_vector])
    idf = np.log10(len(index.DL)/df)
    Q = tf*idf
    Q = Q/np.linalg.norm(Q)
    return Q

def cosine_similarity(D,Q, doc_ids, unique_terms):
    """
    Calculate the cosine similarity for each candidate document in D and a given query (e.g., Q).
    Generate a dictionary of cosine similarity scores 
    key: doc_id
    value: cosine similarity score
    Parameters:
    -----------
    D: DataFrame of tfidf scores.

    Q: vectorized query with tfidf scores
    """

    cos_sim = {}
    for doc_id in doc_ids:
        sim = 0
        for num in range(len(unique_terms)):
            term = unique_terms[num]
            key = (doc_id,term)
            if key in D:
                sim += Q[num]*D[(doc_id,term)]
        cos_sim[doc_id] = sim
    return cos_sim



def get_top_n(sim_dict,N=3):
    return sorted([(doc_id, score) for doc_id, score in sim_dict.items()], key = lambda x: x[1],reverse=True)[:N]


def get_top_n_final(sim_dict, N=3):
    top_n = get_top_n(sim_dict, N)
    return [(doc_id, title_dict[doc_id]) for doc_id, score in top_n]



def search_body_backend(query,index,N=40):

    unique_terms = list(filter(lambda x: x in index.df, tokenize(query)))
    if len(unique_terms) == 0:
        return []
    D, doc_ids = generate_document_tfidf_matrix(unique_terms,index)
    Q = generate_query_tfidf_vector(unique_terms, index)
    top_n = get_top_n_final(cosine_similarity(D,Q, doc_ids, unique_terms),N)
    return top_n



    
def Uniques_from_index(query, index, N=None):

    candidates_and_scores = dict()
    query_words = set(tokenize(query))
    for term in query_words:
        if term in index.df:
            list_of_tuples = read_posting_list(index, term)
            for doc_id, tf in list_of_tuples:
                if doc_id in candidates_and_scores:
                    candidates_and_scores[doc_id] += 1
                else:
                    candidates_and_scores[doc_id] = 1
    return get_top_n_final(candidates_and_scores, N)



def read_posting_list_combined(inverted, w, k, b1, dl_avg, tfidf_weight, unique_weight, bm_weight, final_dict, q_tfidf,
                               terms_len):
    n_ti = inverted.df[w]
    idf_bm = np.log(1 + (len(inverted.DL) - n_ti + 0.5) / (n_ti + 0.5))*bm_weight*(k + 1)
    idf_tfidf = np.log10(len(inverted.DL) / np.array(inverted.df[w])) * q_tfidf * tfidf_weight
    denom_part = k - k * b1
    locs = inverted.posting_locs[w]
    with closing(MultiFileReader()) as reader:
        b = reader.read(locs, inverted.df[w] * TUPLE_SIZE, inverted.folder)
    for i in range(inverted.df[w]):
        doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
        dl = inverted.DL[doc_id]
        freq = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
        tf = freq / dl
        tfidf = tf * idf_tfidf / inverted.vec_norm[doc_id]
        numerator = idf_bm * freq
        denominator = freq + denom_part + k * b1 * dl / dl_avg
        bm = numerator / denominator
        unique = unique_weight / terms_len
        final_dict.update({doc_id: tfidf + bm + unique})
    return

def search_backed_old(query, indices, weights, N, text_dl_avg, title_dl_avg ):



    final_dict = Counter()

    text_tfidf_weight, text_unique_weight, title_tfidf_weight, title_unique_weight, bm_body_weight, bm_title_weight = weights

    cur_index = indices["body"]

    unique_terms = list(filter(lambda x: x in cur_index.df, query))
    wrong_terms = list([word for word in query if word not in unique_terms])
    for term in wrong_terms:
        temp = [(jaccard_distance(set(ngrams(term, 2)), set(ngrams(w, 2))), w) for w in cur_index.df if
                w[0] == term[0]]
        best = sorted(temp, key=lambda val: val[0])[0]
        if best[0] < 0.5:
            unique_terms.append(best[1])
    unique_terms = sorted(unique_terms)

    Q = generate_query_tfidf_vector(unique_terms, cur_index)

    terms_len = len(unique_terms)

    for num in range(terms_len):
        term = unique_terms[num]
        if term in cur_index.df:
            read_posting_list_combined(cur_index, term, 4, 0.5, text_dl_avg, text_tfidf_weight,
                                       text_unique_weight, bm_body_weight, final_dict, Q[num], terms_len)
    cur_index = indices["title"]
    unique_terms = list(filter(lambda x: x in cur_index.df, query))
    wrong_terms = list([word for word in query if word not in unique_terms])
    for term in wrong_terms:
        temp = [(jaccard_distance(set(ngrams(term, 2)), set(ngrams(w, 2))), w) for w in cur_index.df if
                w[0] == term[0]]
        best = sorted(temp, key=lambda val: val[0])[0]
        if best[0] < 0.5:
            unique_terms.append(best[1])
    unique_terms = sorted(unique_terms)

    Q = generate_query_tfidf_vector(unique_terms, cur_index)

    terms_len = len(unique_terms)
    for num in range(terms_len):
        term = unique_terms[num]
        if term in cur_index.df:
            read_posting_list_combined(cur_index, term, 0.2, 1, title_dl_avg, title_tfidf_weight,
                                       title_unique_weight, bm_title_weight, final_dict, Q[num], terms_len)

    return get_top_n_final(final_dict, N)


def search_backend(query, indices, text_dl_avg, title_dl_avg,  weights=[0.70,1.78, 0.54,1.47,0.031,0.072], N=5):

    stemmed_query = [stemmer.stem(token.group()) for token in RE_WORD.finditer(query.lower()) if
             token.group() not in all_stopwords]

    query = tokenize(query)

    final_dict = Counter()

    text_tfidf_weight, text_unique_weight, title_tfidf_weight, title_unique_weight, bm_body_weight, bm_title_weight = weights

    cur_index = indices["body"]

    unique_terms = list(filter(lambda x: x in cur_index.df, query))

    wrong_terms = list([word for word in query if word not in unique_terms])
    for term in wrong_terms:
        temp = [(jaccard_distance(set(ngrams(term, 2)), set(ngrams(w, 2))), w) for w in cur_index.df if
                w[0] == term[0]]
        print(sorted(temp, key=lambda val: val[0]))
        print(term)
        best = sorted(temp, key=lambda val: val[0])[0]
        if best[0] < 0.5:
            unique_terms.append(best[1])
    unique_terms = sorted(unique_terms)

    Q = generate_query_tfidf_vector(unique_terms, cur_index)

    terms_len = len(unique_terms)

    for num in range(terms_len):
        term = unique_terms[num]
        if term in cur_index.df:
            read_posting_list_combined(cur_index, term, 4, 0.1, text_dl_avg, text_tfidf_weight,
                                       text_unique_weight, bm_body_weight, final_dict, Q[num], terms_len)
    cur_index = indices["stem_title"]

    unique_terms = list(filter(lambda x: x in cur_index.df, stemmed_query))
    wrong_terms = list([word for word in stemmed_query if word not in unique_terms])
    for term in wrong_terms:
        temp = [(jaccard_distance(set(ngrams(term, 2)), set(ngrams(w, 2))), w) for w in cur_index.df if
                w[0] == term[0]]
        best = sorted(temp, key=lambda val: val[0])[0]
        if best[0] < 0.5:
            unique_terms.append(best[1])
    unique_terms = sorted(unique_terms)

    Q = generate_query_tfidf_vector(unique_terms, cur_index)

    terms_len = len(unique_terms)
    for num in range(terms_len):
        term = unique_terms[num]
        if term in cur_index.df:
            read_posting_list_combined(cur_index, term, 0.1, 1, title_dl_avg, title_tfidf_weight,
                                       title_unique_weight, bm_title_weight, final_dict, Q[num], terms_len)

    return get_top_n_final(final_dict, N)
