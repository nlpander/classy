import os

import time
import ray
import spacy
import pandas as pd
import re
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords


def WebItemsFilter(text):
    text = text.lower()

    no_auth = 0
    no_em = 0
    no_tw = 0
    no_copy = 0
    no_htmltag = 0

    try:
        auth = re.findall('(by .*;)', text)
        text = text.replace(auth[0], '')
    except:
        no_auth = 1

    try:
        em = re.findall('(\w+@\w+.com)', text)
        text = text.replace(em[0], '')
    except:
        no_em = 1

    try:
        tw = re.findall('(\s@\w+)', text)
        text = text.replace(tw[0], '')
    except:
        no_tw = 1

    try:
        copyright = re.findall('copyright \d+ \w+. all rights reserved.', text)
        text = text.replace(copyright[0], '')
    except:
        no_copy = 1

    # removing html tags
    try:
        tags = re.findall('<[^>]*>', text)
        for tag in tags:
            text = text.replace(tag, '')
    except:
        no_htmltag = 1

    return text


def NumericalExpressionFilter(word_list):
    # This also removes words that contain *any number*
    # return [w for w in word_list if not any(c.isdigit() for c in w)]
    for i in range(0, len(word_list)):

        # current word
        word = word_list[i]

        # currency
        currency = re.findall('(gbp(\d+))|(usd(\d+))|(eur(\d+))', word)

        # time match
        time = re.findall('(\d+:\d+)', word)

        # time period
        period = re.findall('(\d+-day)|(\d+-week)|(\d+-month)|(\d+-year)', word)

        # listed notation
        listed = re.findall('\d+-listed', word)

        # percentage/points
        perc = re.findall('\d+[p]', word)

        # xl then a number
        mult = re.findall('[xl]\d+', word)

        # thousands
        thou = re.findall('\d+,\d+', word)

        # dec
        dec = re.findall('\d+.\d+', word)

        if word.isdigit() or len(thou) != 0 or len(dec) != 0:
            word_list[i] = '#NUM'

        if len(currency) != 0:
            word_list[i] = '#CURRENCY'

        if len(perc) != 0:
            word_list[i] = '#PERC'

        if len(mult) != 0:
            word_list[i] = '#MOD'

        if len(time) != 0:
            word_list[i] = '#TIME'

        if len(period) != 0:
            word_list[i] = '#TIMEPERIOD'

        if len(listed) != 0:
            word_list[i] = '#LISTING'

    return word_list


class Treebank_WordTokenize:
    def __init__(self):
        self.word_tokenizer = TreebankWordTokenizer()
        self.stop_words = stopwords.words('english')

    def transform(self, text):
        words = []
        for w in self.word_tokenizer.tokenize(text):
            w = w.lower()
            if w not in self.stop_words:
                if w.isnumeric():
                    words = words + ['#NUM']
                elif w not in self.stop_words:
                    words = words + [w]

        return words


class Spacy_Tokenize:
    def __init__(self):
        self.tokenizer = spacy.load('en_core_web_sm', disable=['tok2vec', 'attribute_ruler', 'lemmatizer'])

    def transform(self, text):
        words = []
        for w in self.tokenizer(text):
            if ((w.is_stop is False) and (w.like_email is False)) and (w.like_url is False):
                if w.ent_type_ != '':
                    words = words + [w.ent_type_]
                else:
                    words = words + [w.text.lower()]

        return words


def FilterTokenize_Job(df: pd.DataFrame) -> pd.DataFrame:
    mode = df['mode'].values[0]
    text_col = df['text_col'].values[0]

    # remove html tags / emails / websites
    df.loc[:, text_col] = df[text_col].apply(lambda x: WebItemsFilter(x))

    # tokenize the text
    if mode == 'treebank':

        tokenizer = Treebank_WordTokenize()
        df.loc[:, 'tokens'] = df[text_col].apply(lambda x: tokenizer.transform(x))

        # translate numerical expressions
        df.loc[:, 'tokens'] = df.tokens.apply(lambda x: NumericalExpressionFilter(x))

    elif mode == 'spacy':

        tokenizer = Spacy_Tokenize()
        df.loc[:, 'tokens'] = df[text_col].apply(lambda x: tokenizer.transform(x))

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
    tokens_df = tr_ds.map(lambda x: x[["tokens"]]).to_pandas(limit=n_rows).rename({'value': 'tokens'}, axis=1)

    # remove datasets and temporary dataframe in memory
    del tr_ds, tmp_df
    ray.shutdown()
    import gc
    gc.collect()

    # concatenate original dataframe and tokens dataframe
    df = pd.concat([df, tokens_df], axis=1)

    print('Total Time Elapsed: %.2f' % (time.time() - t0))

    return df
