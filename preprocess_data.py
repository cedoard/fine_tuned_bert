import re

from ekphrasis.classes.preprocessor import TextPreProcessor
import numpy as np
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

from bert_repo.run_classifier import InputExample


def ekphrasis_preprocess(s):
    text_processor = TextPreProcessor(
        # terms that will be normalized
        normalize=['url', 'email', 'user', 'percent', 'money', 'phone', 'time', 'date', 'number'],
        # terms that will be annotated
        annotate={"hashtag"},
        fix_html=True,  # fix HTML tokens

        unpack_hashtags=True,  # perform word segmentation on hashtags

        # select a tokenizer. You can use SocialTokenizer, or pass your own
        # the tokenizer, should take as input a string and return a list of tokens
        tokenizer=SocialTokenizer(lowercase=True).tokenize,
        dicts=[emoticons]
    )

    s = s.lower()
    s = str(" ".join(text_processor.pre_process_doc(s)))
    s = re.sub(r"[^a-zA-ZÀ-ú</>!?♥♡\s\U00010000-\U0010ffff]", ' ', s)
    s = re.sub(r"\s+", ' ', s)
    s = re.sub(r'(\w)\1{2,}', r'\1\1', s)
    s = re.sub(r'^\s', '', s)
    s = re.sub(r'\s$', '', s)
    return s


def convert_single_string_to_input_dict(tokenizer, example_string, max_seq_length):

    token_a = tokenizer.tokenize(example_string)

    tokens = []
    segments_ids = []
    segment_ids = []

    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in token_a:
        tokens.append(token)
        segment_ids.append(0)

    tokens.append('[SEP]')
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    label_id = [0]
    padding = [0] * max_seq_length
    print(len(input_ids), len(input_mask), len(segment_ids), len(label_id))
    return {"input_ids": [input_ids, padding], "input_mask": [input_mask, padding],
            "segment_ids": [segment_ids, padding], "label_ids": label_id}


