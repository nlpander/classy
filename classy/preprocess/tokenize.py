import os

import time
import ray
import spacy
import pandas as pd
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords

from classy.preprocess.filters import WebItemsFilter, NumericalExpressionFilter


class Treebank_WordTokenize:
    def __init__(self):
        self.word_tokenizer = TreebankWordTokenizer()
        self.stop_words = stopwords.words('english')

        neg_words = [x for x in self.stop_words if 'n\'t' in x or x in ['no', 'not', 'nor']]
        neg_words = neg_words + ['ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn',
                                 'haven', 'isn', 'mightn', 'mustn', 'needn', 'shan',
                                 'shouldn', 'wasn', 'weren', 'won', 'wouldn']
        pol_words = ['up', 'down', 'above', 'below', 'over', 'under', 'on', 'off', 'few', 'more',
                     'most', 'very', 'during', 'through', 'against', 'until', 'do', 'doing', 'did',
                     'does', 'before', 'after', 'between']
        mod_words = ['should', 'could', 'would']

        self.stop_words = [x for x in self.stop_words if x not in (neg_words + pol_words + mod_words)]
        self.neg_words = neg_words
        self.mod_words = mod_words

    def transform(self, text):
        words = []
        for w in self.word_tokenizer.tokenize(text):
            w = w.lower()
            if w in self.stop_words:
                pass
            elif w in self.neg_words:
                words.append('#NEG')
            elif w in self.mod_words:
                words.append('#MOD')
            elif w.isnumeric():
                words.append('#NUM')
            else:
                words.append(w)

        return words


class Spacy_Tokenize:
    def __init__(self):
        self.tokenizer = spacy.load('en_core_web_sm', disable=['tok2vec', 'attribute_ruler', 'lemmatizer'])

    def transform(self, text):
        words = []
        for w in self.tokenizer(text):
            if w.is_stop or w.like_email or w.like_url:
                pass
            elif w.ent_type_ != '':
                words.append(w.ent_type_)
            else:
                words.append(w.text.lower())

        return words


def FilterTokenize_Job(df: pd.DataFrame) -> pd.DataFrame:
    mode = df['mode'].values[0]
    text_col = df['text_col'].values[0]

    # remove html tags / emails / websites
    df.loc[:, text_col] = df[text_col].apply(WebItemsFilter)

    # tokenize the text
    if mode == 'treebank':

        tokenizer = Treebank_WordTokenize()
        df.loc[:, 'tokens'] = df[text_col].apply(tokenizer.transform)

        # translate numerical expressions
        df.loc[:, 'tokens'] = df.tokens.apply(NumericalExpressionFilter)

    elif mode == 'spacy':

        tokenizer = Spacy_Tokenize()
        df.loc[:, 'tokens'] = df[text_col].apply(tokenizer.transform)

    return df


def FilterTokenize_DF(df, text_col='body', mode='treebank', output_col='tokens', workers=os.cpu_count()):
    t0 = time.time()
    df.loc[df.index, 'text_col'] = text_col
    df.loc[df.index, 'mode'] = mode
    df.loc[df.index, 'output_col'] = output_col
    tmp_df = df[['text_col', 'mode', text_col]]

    # convert to a ray distributed dataset and partition dataframe
    ds = ray.data.from_pandas(tmp_df)
    ds = ds.repartition(workers)

    # perform filtering and tokenization
    tr_ds = ds.map_batches(FilterTokenize_Job)

    # extract the tokens column and convert ds to dataframe
    n_rows = tr_ds.count()
    tokens_df = tr_ds.map_batches(lambda x: x[["tokens"]]).to_pandas(limit=n_rows).rename({'value': 'tokens'}, axis=1)

    # remove datasets and temporary dataframe in memory
    del tr_ds, tmp_df
    ray.shutdown()
    import gc
    gc.collect()
    # drop parameter columns
    df = df.drop(['text_col', 'mode', 'output_col'], axis=1)

    # concatenate original dataframe and tokens dataframe
    df = pd.concat([df, tokens_df], axis=1)

    print('Total Time Elapsed: %.2f' % (time.time() - t0))

    return df
