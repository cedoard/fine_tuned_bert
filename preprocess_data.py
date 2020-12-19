from ekphrasis.classes.preprocessor import TextPreProcessor
import numpy as np
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

from bert_repo.run_classifier import InputExample


def preprocessing_data(sentences, labels):


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

    # PREPROCESS TRAINING AND TEST DATA
    def func(row):
        if row == 'POS':
            return 1
        elif row == 'NEG':
            return 0

    # final examples training
    sentences_filtered = []
    labels = list(map(lambda x: func(x), labels))

    import re
    i = 0
    for s in sentences:
        s = s.lower()
        s = str(" ".join(text_processor.pre_process_doc(s)))
        s = re.sub(r"[^a-zA-ZÀ-ú</>!?♥♡\s\U00010000-\U0010ffff]", ' ', s)
        s = re.sub(r"\s+", ' ', s)
        s = re.sub(r'(\w)\1{2,}', r'\1\1', s)
        s = re.sub(r'^\s', '', s)
        s = re.sub(r'\s$', '', s)
        # print(s)
        sentences_filtered.append([labels[i], s])
        i = i + 1

    sentences_filtered = np.array(sentences_filtered)
    np.random.shuffle(sentences_filtered)
    split = int(len(sentences_filtered) * 0.8)

    sentences_filtered_train, sentences_filtered_test = sentences_filtered[:split], sentences_filtered[split:]

    f = lambda x: InputExample(guid=None,  # Globally unique ID for bookkeeping, unused in this example
                               text_a=x[1],
                               text_b=None,
                               label=int(x[0]))

    f2 = lambda x: InputExample(guid=None,  # Globally unique ID for bookkeeping, unused in this example
                                text_a=x[1],
                                text_b=None,
                                label=0)

    train_examples = map(f, sentences_filtered_train)
    train_examples = list(train_examples)
    train_examples = np.array(train_examples)

    test_examples = map(f2, sentences_filtered_test)
    test_examples = list(test_examples)
    test_examples = np.array(test_examples)

    return train_examples, test_examples