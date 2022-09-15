import os
import time
import numpy as np
import pandas as pd
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument


def token2ind(tokens, index2word):
    ind = []
    N = len(index2word) + 1

    for word in tokens:
        if word in index2word.keys():
            ind.append(index2word[word])
        else:
            ind.append(N)

    return ind


def zero_pad(x, max_len):
    d = max_len - len(x)

    if d > 0:
        return x + d * [0]
    else:
        return x[0:max_len]


def read_corpus_d2v(df):
    for i in range(0, len(df)):
        yield TaggedDocument(df['tokens'].loc[i], [df['TAG'].loc[i]])


class Generate_Embeddings:

    def __init__(self,
                 embedding='w2v',
                 word_window=5,
                 min_count=30,
                 embedding_dimension=100,
                 threshold_mode='perc',
                 seq_len_threshold_perc=0.95,
                 seq_len_hard_threshold=300,
                 workers=os.cpu_count()):

        self.embedding = embedding
        self.word_window = word_window
        self.min_count = min_count
        self.embedding_dimension = embedding_dimension
        self.threshold_mode = threshold_mode
        self.seq_len_threshold_perc = seq_len_threshold_perc
        self.seq_len_hard_threshold = seq_len_hard_threshold
        self.workers = workers
        self.threshold = None
        self.embedding_matrix = None
        self.tokens = None

    def compute(self, df_):

        ######## training embedding model ########

        T0 = time.time()

        if self.embedding == 'w2v':

            sentences = list(df_['tokens'].values)

            embedding_model = Word2Vec(window=self.word_window,
                                       min_count=self.min_count,
                                       vector_size=self.embedding_dimension,
                                       workers=self.workers)

        elif self.embedding == 'd2v':

            sentences = [TaggedDocument(x[0], [x[1]]) for _, x in enumerate(df_.loc[:, ['tokens', 'TAG']].values)]

            embedding_model = Doc2Vec(window=self.word_window,
                                      min_count=self.min_count,
                                      vector_size=self.embedding_dimension,
                                      workers=self.workers)

        embedding_model.build_vocab(sentences)

        vsz = len(embedding_model.wv.index_to_key)
        print('Vocabulary Size: %d' % (vsz))
        print('')

        t0 = time.time()
        embedding_model.train(sentences,
                              total_examples=len(sentences),
                              total_words=vsz, epochs=30)

        print('%s trained: %.3f s Elapsed' % (self.embedding, time.time() - t0))
        print('')

        ######## do length thresholding ########

        if self.threshold_mode == 'perc':
            self.threshold = int(np.ceil(np.exp(df_.tokens.apply(lambda x: np.log(len(x) + 1)). \
                                                quantile(self.seq_len_threshold_perc)) - 1))
        elif self.threshold_mode == 'hard':
            self.threshold = self.seq_len_hard_threshold
        else:
            self.threshold = 300
            print('No percentile threshold or hard threshold for sequence length \
            specified specified default length of %d' % self.threshold)

        print('Sequence Length Threshold: %d' % self.threshold)
        print('')

        ######## convert word tokens to indices ########

        if self.embedding == 'w2v':

            index2word = embedding_model.wv.key_to_index

            # get the sequence of indices based on each token
            df_['ind'] = df_['tokens'].apply(lambda x: token2ind(x, index2word))

            # do zero padding
            df_.loc[df_.index, 'ind_pad'] = df_.loc[df_.index, 'ind'].apply(
                lambda x: np.array(zero_pad(x, self.threshold)))

            # construct embedding matrix
            self.embedding_matrix = np.vstack((np.zeros((1, self.embedding_dimension)),
                                               embedding_model.wv.vectors,
                                               np.random.rand(self.embedding_dimension)))

        elif self.embedding == 'd2v':

            # adding indices for the paragraph identities
            df_['tokens'] = df_['TAG'].apply(lambda x: ['#TAG' + str(int(x))]) + df_['tokens']
            tag_labels = list(df_.TAG.unique())
            NT = len(tag_labels)

            index2word = {embedding_model.wv.index_to_key[i]: i + 1 + NT
                          for i in range(0, len(embedding_model.wv.index_to_key))}

            for tag in tag_labels:
                index2word['#TAG%d' % tag] = tag

            # get the sequence of indices based on each token
            df_['ind'] = df_['tokens'].apply(lambda x: token2ind(x, index2word))

            # do zero padding
            df_.loc[df_.index, 'ind_pad'] = df_.loc[df_.index, 'ind'].apply(
                lambda x: np.array(zero_pad(x, self.threshold)))

            # construct embedding matrix
            self.embedding_matrix = np.vstack((np.zeros((1, self.embedding_dimension)),
                                               embedding_model.dv.vectors,
                                               embedding_model.wv.vectors,
                                               np.random.rand(self.embedding_dimension)))

        print('embedding matrix dimensions:')
        print(self.embedding_matrix.shape)
        print('')

        self.tokens = np.array([x for x in df_.ind_pad.values])
        print('token dimensions:')
        print(self.tokens.shape)
        print('')

        print('Embeddings, Tokens and Labels Generated... %.3f s Elapsed' % (time.time() - T0))

        return [self.embedding_matrix, self.tokens]

    def get_embedding_matrix(self):
        return self.embedding_matrix

    def get_tokens(self):
        return self.tokens


def EmbeddingModel_JobRay(df: pd.DataFrame) -> list:
    ge = Generate_Embeddings(word_window=5, min_count=30, embedding_dimension=100,
                             threshold_mode='perc', seq_len_threshold_perc=0.95,
                             workers=1)
    output = ge.compute(df)

    return output
