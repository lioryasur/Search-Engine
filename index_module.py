import pyspark
import sys
import os
from time import time, strftime, gmtime
from google.cloud import storage
import tqdm
from inverted_index_gcp import InvertedIndex
import random
random.seed = 42
import numpy as np
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf, SparkFiles
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
import pandas as pd
import pyarrow.parquet as pq
import pickle

# Put your bucket name below and make sure you can access it without an error



def create_title_dict():
    loaded_table = pickle.load(open("table.pickle", "rb"))
    return loaded_table



def load_body_index():
    text_folder_dst = 'text_postings_fixed'
    # text_src = f'gs://{bucket_name}/{text_folder_dst}/'
    # !gsutil -m cp -r $text_src "./"
    text_inverted_fixed = InvertedIndex().read_index('text_postings_fixed/', 'text_index')
    text_inverted_fixed.folder = text_folder_dst
    return text_inverted_fixed

def load_anchor_index():
    text_folder_dst = 'anchor_postings'
    # text_src = f'gs://{bucket_name}/{text_folder_dst}/'
    # !gsutil -m cp -r $text_src "./"
    anchor_inverted_fixed = InvertedIndex().read_index('anchor_postings/', 'anchor_index')
    anchor_inverted_fixed.folder = text_folder_dst
    return anchor_inverted_fixed


def load_title_index():
    title_folder_dst = 'title_postings_fixed'
    # title_src = f'gs://{bucket_name}/{title_folder_dst}/'
    # !gsutil -m cp -r $title_src "./"
    title_inverted_fixed = InvertedIndex().read_index('title_postings_fixed/', 'title_index')
    title_inverted_fixed.folder = title_folder_dst
    return title_inverted_fixed

def load_stem_title_index():
    stem_title_folder_dst = 'title_postings_stem'
    # stem_title_src = f'gs://{bucket_name}/{title_folder_dst}/'
    # !gsutil -m cp -r $title_src "./"
    stem_title_inverted_fixed = InvertedIndex().read_index('title_postings_stem/', 'title_index')
    stem_title_inverted_fixed.folder = stem_title_folder_dst
    return stem_title_inverted_fixed


def average_precision(true_list, predicted_list, k=40):
    true_set = frozenset(true_list)
    predicted_list = predicted_list[:k]
    precisions = []
    for i,doc_id in enumerate(predicted_list):
        if doc_id in true_set:
            prec = (len(precisions)+1) / (i+1)
            precisions.append(prec)
    if len(precisions) == 0:
        return 0.0
    return np.round(np.sum(precisions)/len(precisions),3)