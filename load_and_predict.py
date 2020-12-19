import sys

import pandas as pd
import numpy as np

from tensorflow.contrib import predictor

if not 'bert_repo' in sys.path:
  sys.path += ['bert_repo']
from bert_repo.run_classifier import *
import bert_repo.modeling
import bert_repo.optimization
from bert_repo import tokenization



from preprocess_data import preprocessing_data

MAX_SEQ_LENGTH = 128
LOAD_PATH = 'trained_model/1608370941'
VOCAB_FILE = 'data/vocab.txt'

label_list = [0, 1]
#Inizialize BERT tokenizer
tokenizer = tokenization.FullTokenizer(VOCAB_FILE, do_lower_case=True)


#LOAD TRAINING AND TEST DATA
training_data = pd.read_excel('data/rev_df_final.xlsx', engine='openpyxl')
training_data = training_data.loc[~training_data.sentiment.isin(['NEUTRAL'])]
training_data = training_data.dropna().reset_index(drop=True)
sentences = training_data.iloc[:, 0]
labels = training_data.iloc[:, -1]
sentences = training_data.iloc[:, 0]
labels = training_data.iloc[:, -1]

train_examples, test_examples = preprocessing_data(sentences, labels)

#CONVERT DATA TO FEATURES
input_features_test = convert_examples_to_features( test_examples, label_list, max_seq_length=128, tokenizer=tokenizer)
print(test_examples[0:10])

#LOAD_MODEL
predict_fn = predictor.from_saved_model(LOAD_PATH)

#PREDICT
predictions = predict_fn({'example':input_features_test[0]})