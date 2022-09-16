import pandas as pd
import time
from classy.tokenize_tf import FilterTokenize_DF
from classy.emb_tf import Generate_Embeddings

if __name__ == '__main__':

    path = 'news_sample.csv';
    df = pd.read_csv(path, encoding_errors='ignore', index_col=False,nrows=1e5);

    t0 = time.time()
    df_ = FilterTokenize_DF(df, text_col='body', mode='treebank',output_col='tokens', workers=8)
    print('Tokenization Time Elapsed : %.3f' % (time.time() - t0))

    t0 = time.time()
    ge = Generate_Embeddings(embedding='w2v', workers=8, token_column='tokens')
    embedding_matrix, index2word, tokens = ge.compute_embedding(df_)
    print('Embedding Generation Time Elapsed: %.3f' % (time.time() - t0))