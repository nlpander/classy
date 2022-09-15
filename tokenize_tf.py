import spacy
import pandas as pd
import re
from nltk.tokenize import TreebankWordTokenizer


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


class Treebank_WordTokenize():
    def __init__(self):
        self.word_tokenizer = TreebankWordTokenizer()
        self.stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
                           'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
                           'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
                           'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
                           'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                           'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
                           'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
                           'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
                           'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
                           'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                           'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
                           'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
                           'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
                           'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now',
                           'd', 'll', '\'ll', 'm', 'o', 're', 've', '\'ve', 'y', 'ain', 'aren', 'couldn', 'didn',
                           'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn',
                           'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn', '\'s', '\'re']

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


class Filter_Tokenize:

    def __init__(self, mode='treebank', text_column='body'):

        self.mode = mode
        self.tokenizer = None
        self.text_col = text_column

    def transform(self, df):

        # remove html tags / emails / websites
        df.loc[:, self.text_col] = df[self.text_col].apply(lambda x: WebItemsFilter(x))

        # tokenize the text
        if self.mode == 'treebank':

            self.tokenizer = Treebank_WordTokenize()
            df.loc[:, 'tokens'] = df.loc[:, self.text_col].apply(lambda x: self.tokenizer.transform(x))

            # translate numerical expressions
            df.loc[:, 'tokens'] = df.tokens.apply(lambda x: NumericalExpressionFilter(x))

        elif self.mode == 'spacy':

            self.tokenizer = Spacy_Tokenize()
            df.loc[:, 'tokens'] = df.loc[:, self.text_col].apply(lambda x: self.tokenizer.transform(x))

        return df


def SpacyTokenize_JobRay(df: pd.DataFrame) -> pd.DataFrame:
    spt = Spacy_Tokenize()
    df.loc[:, 'tokens'] = df.text.apply(lambda x: spt.transform(x))

    return df


def TreebankTokenize_JobRay(df: pd.DataFrame) -> pd.DataFrame:
    twt = Treebank_WordTokenize()
    df.loc[:, 'tokens'] = df.text.apply(lambda x: twt.transform(x))

    return df