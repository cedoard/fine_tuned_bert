import sys
import warnings
from pprint import pprint

warnings.simplefilter(action='ignore')

import pandas as pd
import numpy as np

from tensorflow.contrib import predictor

if not 'bert_repo' in sys.path:
    sys.path += ['bert_repo']

from bert_repo.run_classifier import *
import bert_repo.modeling
import bert_repo.optimization
from bert_repo import tokenization

from preprocess_data import convert_single_string_to_input_dict, ekphrasis_preprocess

#params
MAX_SEQ_LENGTH = 128
LOAD_PATH = 'trained_model/1608503656'
VOCAB_FILE = 'data/vocab.txt'
label_list = [0, 1]

# Inizialize BERT tokenizer
tokenizer = tokenization.FullTokenizer(VOCAB_FILE, do_lower_case=True)

# LOAD TRAINING AND TEST DATA
training_data = pd.read_excel('data/rev_df_final.xlsx', engine='openpyxl')
training_data = training_data.loc[~training_data.sentiment.isin(['NEUTRAL'])]
training_data = training_data.dropna().reset_index(drop=True)
sentences = training_data.iloc[:, 0]
labels = training_data.iloc[:, -1]
sentences = training_data.iloc[:, 0]
labels = training_data.iloc[:, -1]

# train_examples, test_examples = preprocessing_data(sentences, labels)

# LOAD_MODEL
predict_fn = predictor.from_saved_model(LOAD_PATH)

# INPUT_SENTENCES
example_sent_neg = "brutto e cattivo, sono veramente triste mi vorrei uccidere la mia vita non ha senso è terribile " \
                   "male male "
example_sent_pos = "sono euforico, mi piace così tanto che sono felice solo di poter essere vivo e poter prendere il " \
                   "monopattino per raggiungere l'apice della mia felicità "

example_sent_neutral = "Il film non è bello, però dopo la prima parte mi è piaciuto moltissimo"

def predict(tokenizer, predict_fn, input_str, MAX_SEQ_LENGTH):
    # CONVERT DATA TO FEATURES
    example_prep = ekphrasis_preprocess(input_str)
    example_features = convert_single_string_to_input_dict(tokenizer=tokenizer,
                                                       example_string=example_prep,
                                                       max_seq_length=MAX_SEQ_LENGTH)

    prediction = predict_fn(example_features)['probabilities'][0]
    prediction_dict = {'POS': round(prediction[1],4), 'NEG': round(prediction[0],4)}
    pprint(f"prediction: {prediction_dict}")
    return prediction


# PREDICT
predict(tokenizer, predict_fn, example_sent_neutral, MAX_SEQ_LENGTH)
