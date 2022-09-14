from gensim.models import Word2Vec,Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import pandas as pd
import numpy as np
import time, os

def token2ind(tokens, index2word):
    
    ind = []
    N = len(index2word) + 1
    
    for word in tokens:
        if word in index2word.keys():
            ind.append(index2word[word])
        else:
            ind.append(N)
        
    return ind

def zero_pad(x,max_len):
    
    d = max_len - len(x)
    
    if d > 0:
        return x + d*[0]
    else:
        return x[0:max_len]
    
def read_corpus(df):
    for i in range(0,len(df)):
        yield TaggedDocument(df['tokens'].loc[i],[df['TAG'].loc[i]])

class Generate_Embeddings():

    def __init__(self,word_window=5, min_count=30, embedding_dimension=100,\
                 threshold_mode='perc',seq_len_threshold_perc=0.95,\
                 seq_len_hard_threshold=300,workers=os.cpu_count()):
        
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
        
    def compute(self,df_):
        
        T0 = time.time()
        
        sentences = list(df_['tokens'].values)
        w2v = Word2Vec(window=self.word_window,min_count=self.min_count,\
                       vector_size=self.embedding_dimension,\
                       workers=self.workers)

        w2v.build_vocab(sentences)

        vsz = len(w2v.wv.index_to_key)
        print('Vocabulary Size: %d'%(vsz))
        print('')
        
        t0 = time.time()
        w2v.train(sentences,total_examples=len(sentences),total_words=vsz,epochs=30)
        print('Word 2 Vec trained: %.3f s Elapsed'%(time.time()-t0))
        print('')
        
        
        if self.threshold_mode == 'perc':
            self.threshold = int(np.ceil(np.exp(df_.tokens.apply(lambda x:np.log(len(x)+1)).\
                                            quantile(self.seq_len_threshold_perc)) - 1))
        elif self.threshold_mode == 'hard':
            self.threshold = self.seq_len_hard_threshold
        else:
            self.threshold = 300
            print('No percentile threshold or hard threshold for sequence length \
            specified specified default length of %d'%self.threshold)
        
        print('Sequence Length Threshold: %d'%self.threshold)
        print('')
        
        df_['ind'] = df_['tokens'].apply(lambda x:token2ind(x,w2v.wv.key_to_index))
        df_.loc[df_.index,'ind_pad'] = df_.loc[df_.index,'ind'].apply(lambda x:np.array(zero_pad(x,self.threshold)))

        self.embedding_matrix = np.vstack((np.zeros((1,self.embedding_dimension)),\
                                w2v.wv.vectors,\
                                np.random.rand((self.embedding_dimension))))

        print('embedding matrix dimensions:')
        print(self.embedding_matrix .shape)
        print('')

        self.tokens = np.array([x for x in df_.ind_pad.values])
        print('token dimensions:')
        print(self.tokens.shape)
        print('')
        
        print('Embeddings, Tokens and Labels Generated... %.3f s Elapsed'%(time.time()-T0))
    
        return [self.embedding_matrix,self.tokens]
        
    def get_embedding_matrix(self):
        return self.embedding_matrix
    
    def get_tokens(self):
        return self.tokens 
    
    
def W2VEmbed_JobRay(df: pd.DataFrame) -> list:

    ge = Generate_Embeddings(word_window=5, min_count=30, embedding_dimension=100,\
                            threshold_mode='perc',seq_len_threshold_perc=0.95,\
                            workers=1)
    output = ge.compute(df)
    
    return output