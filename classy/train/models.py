import torch.nn as nn
import torch
import torch.nn.functional as F
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer


class Conv1D_Network(nn.Module):

    def __init__(self, seq_len, embedding_matrix, freeze_embedding=True,
                 maxPool=True, filter_dimensions=(1, 3, 3, 5), number_filters=8,
                 dropout=0.8):
        super().__init__()

        self.seq_len = seq_len
        self.embedding_matrix = embedding_matrix
        self.vocab_size = embedding_matrix.shape[0]
        self.embedding_dim = embedding_matrix.shape[1]
        self.freeze_embedding = freeze_embedding
        self.maxPool = maxPool

        self.filter_dim = filter_dimensions
        self.number_filter_types = len(filter_dimensions)
        self.number_filters = number_filters
        self.dropout = dropout

        self.Embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim, padding_idx=0). \
            from_pretrained(embeddings=self.embedding_matrix, freeze=self.freeze_embedding).to(torch.float32)

        self.ConvLayers = nn.ModuleList()
        for ci in range(0, self.number_filter_types):
            self.ConvLayers.append(nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.number_filters,
                                             kernel_size=self.filter_dim[ci],
                                             padding=int((self.filter_dim[ci] - 1) / 2)).to(torch.float32))

        self.Flatten = nn.Flatten()
        self.Dropout = nn.Dropout(dropout)

        if self.maxPool:
            self.PoolLayer = nn.MaxPool1d(self.number_filter_types * self.number_filters)
            self.OutputLayer = nn.Linear(in_features=self.seq_len, out_features=1, bias=True).to(torch.float32)
        else:
            self.OutputLayer = nn.Linear(in_features=self.seq_len * self.number_filter_types * self.number_filters,
                                         out_features=1, bias=True).to(torch.float32)

    def forward(self, seq):

        y = self.Embedding(seq).transpose(1, 2)
        f = []

        for ci in range(0, len(self.ConvLayers)):
            f.append(self.ConvLayers[ci](y))

        fc = torch.cat(tuple(f), 1)

        if self.maxPool:
            fc = self.PoolLayer(fc.transpose(1, 2))

        fc_ = self.Flatten(fc)
        fc_ = self.Dropout(fc_)

        output = torch.sigmoid(self.OutputLayer(fc_))

        return output


class Conv1D_Network_MultLabel(nn.Module):

    def __init__(self, seq_len, num_labels, embedding_matrix,
                 freeze_embedding=True, maxPool=True,
                 filter_dimensions=(1, 3, 3, 5), number_filters=8,
                 dropout=0.8):
        super().__init__()

        self.seq_len = seq_len
        self.num_labels = num_labels

        self.embedding_matrix = embedding_matrix
        self.vocab_size = embedding_matrix.shape[0]
        self.embedding_dim = embedding_matrix.shape[1]
        self.freeze_embedding = freeze_embedding
        self.maxPool = maxPool

        self.filter_dim = filter_dimensions
        self.number_filter_types = len(filter_dimensions)
        self.number_filters = number_filters
        self.dropout = dropout

        self.Embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim, padding_idx=0). \
            from_pretrained(embeddings=self.embedding_matrix, freeze=self.freeze_embedding).double()

        self.ConvLayers = nn.ModuleList()

        for ci in range(0, self.number_filter_types):
            self.ConvLayers.append(nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.number_filters,
                                             kernel_size=self.filter_dim[ci],
                                             padding=int((self.filter_dim[ci] - 1) / 2)). \
                                   double())

        self.Flatten = nn.Flatten()

        self.Dropout = nn.Dropout(dropout)

        if self.maxPool:
            self.PoolLayer = nn.MaxPool1d(self.number_filter_types * self.number_filters)
            self.OutputLayer = nn.Linear(in_features=self.seq_len, out_features=1, bias=True).double()
        else:
            self.OutputLayer = nn.Linear(in_features=self.seq_len * self.number_filter_types * self.number_filters,
                                         out_features=self.num_labels, bias=True).double()

    def forward(self, seq):

        y = self.Embedding(seq).transpose(1, 2)

        f = []
        for ci in range(0, len(self.ConvLayers)):
            f.append(self.ConvLayers[ci](y))

        fc = torch.cat(tuple(f), 1)

        if self.maxPool:
            fc = self.PoolLayer(fc.transpose(1, 2))

        fc_ = self.Flatten(fc)
        fc_ = self.Dropout(fc_)

        output = torch.sigmoid(self.OutputLayer(fc_))

        return output


class TransformerSelfAttentionHead(nn.Module):

    def __init__(self, seq_len, embed_dim, hidden_units, dropout=0.8):
        super().__init__()

        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.hidden_units = hidden_units
        self.dropout = dropout

        self.WQ = nn.Linear(in_features=embed_dim, out_features=self.hidden_units, bias=True)
        self.WK = nn.Linear(in_features=embed_dim, out_features=self.hidden_units, bias=True)
        self.WV = nn.Linear(in_features=embed_dim, out_features=self.hidden_units, bias=True)

        self.Dropout = nn.Dropout(self.dropout)

    def forward(self, inputA):
        Q = self.WQ(inputA)
        Q = torch.relu(Q)

        # Q = self.Dropout(Q)

        K = self.WK(inputA)
        K = torch.relu(K)

        # K = self.Dropout(K)

        V = self.WV(inputA)
        V = torch.relu(V)

        # V = self.Dropout(V)

        QK = torch.bmm(Q, K.transpose(1, 2))
        QK = self.Dropout(QK)

        QKs = F.softmax(QK / self.embed_dim ** 0.5, dim=1)
        QKs = F.softmax(QKs, dim=2)

        Z = torch.bmm(QKs, V)

        return Z


class TransformerHead(nn.Module):

    def __init__(self, seq_len, embed_dim, hidden_units_sa, hidden_units_out, sa_heads=4, dropout=0.8):
        super().__init__()

        self.seq_len = seq_len
        self.hidden_units_sa = hidden_units_sa
        self.embed_dim = embed_dim
        self.sa_heads = sa_heads
        self.hidden_units_out = hidden_units_out
        self.dropout = dropout

        self.SAHeads = nn.ModuleList()

        for mi in range(0, self.sa_heads):
            self.SAHeads.append(TransformerSelfAttentionHead(seq_len=self.seq_len, embed_dim=self.embed_dim,
                                                             hidden_units=self.hidden_units_sa, dropout=self.dropout))

        self.WZ = nn.Linear(in_features=self.hidden_units_sa * self.sa_heads, out_features=self.hidden_units_out, bias=True)
        self.Dropout = nn.Dropout(self.dropout)

    def forward(self, inputA):

        Z_ = []

        for mi in range(0, self.sa_heads):
            Z_.append(self.SAHeads[mi](inputA))

        Zall = torch.cat(tuple(Z_), dim=2)

        ZF = self.WZ(Zall)
        ZF = self.Dropout(ZF)

        return ZF


class ClassicTransformer(nn.Module):

    def __init__(self, seq_len, hidden_units_sa, hidden_units_out,
                 embedding_matrix, num_labels, sa_heads=4, sa_modules=1,
                 freeze_embedding=True, dropout=0.8, layer_norm=False):
        super().__init__()

        self.seq_len = seq_len
        self.hidden_units_sa = hidden_units_sa
        self.hidden_units_out = hidden_units_out
        self.sa_heads = sa_heads
        self.sa_modules = sa_modules
        self.embedding_matrix = embedding_matrix
        self.num_labels = num_labels
        self.vocab_size = embedding_matrix.shape[0]
        self.embed_dim = embedding_matrix.shape[1]
        self.dropout = dropout
        self.layer_norm = layer_norm

        self.Embedding = torch.nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_dim,
                                            padding_idx=0). \
            from_pretrained(embeddings=embedding_matrix, freeze=freeze_embedding)

        self.TransformerHeads = nn.ModuleList()

        for ti in range(0, self.sa_modules):
            self.TransformerHeads.append(TransformerHead(seq_len=self.seq_len, embed_dim=self.embed_dim,
                                                         hidden_units_sa=int(self.hidden_units_sa / self.sa_heads),
                                                         hidden_units_out=self.hidden_units_out,
                                                         sa_heads=self.sa_heads, dropout=self.dropout))

        self.LN = nn.LayerNorm([self.seq_len, self.embed_dim])

        self.Flatten = nn.Flatten()

        self.Dropout = nn.Dropout(self.dropout)

        self.OutputLayer = nn.Linear(in_features=self.hidden_units_out, out_features=self.num_labels, bias=True)

    def forward(self, seqs):

        X = self.Embedding(seqs)

        ZF = self.TransformerHeads[0](X)

        if self.layer_norm:
            X_ = self.LN(ZF + X)
        else:
            X_ = ZF + X

        if self.sa_modules > 1:

            for ti in range(1, self.sa_modules):
                ZF = self.TransformerHeads[ti](X_)

                if self.layer_norm:
                    X_ = self.LN(ZF + X_)
                else:
                    X_ = ZF + X_

        Xf = self.Flatten(X_)

        Xf = self.Dropout(Xf)

        y = self.OutputLayer(Xf)
        y = torch.sigmoid(y)

        return y


class Conv1D_Network_MultLabel_SA(nn.Module):

    def __init__(self, seq_len,
                 num_labels,
                 embedding_matrix,
                 hidden_units_sa,
                 hidden_units_out,
                 freeze_embedding=True,
                 filter_dimensions=(1, 3, 3, 5),
                 number_filters=8,
                 sa_modules=1,
                 sa_heads=4,
                 layer_norm=True,
                 dropout=0.8):
        super().__init__()

        self.seq_len = seq_len
        self.num_labels = num_labels

        self.embedding_matrix = embedding_matrix
        self.vocab_size = embedding_matrix.shape[0]
        self.embedding_dim = embedding_matrix.shape[1]
        self.freeze_embedding = freeze_embedding

        self.filter_dim = filter_dimensions
        self.number_filter_types = len(filter_dimensions)
        self.number_filters = number_filters
        self.sa_modules = sa_modules
        self.sa_heads = sa_heads
        self.hidden_units_sa = hidden_units_sa
        self.hidden_units_out = hidden_units_out
        self.layer_norm = layer_norm
        self.dropout = dropout

        self.Embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim, padding_idx=0). \
            from_pretrained(embeddings=self.embedding_matrix, freeze=self.freeze_embedding)

        self.PositionEmbedding = PositionalEncoding1D(channels=self.seq_len)
        self.LN = nn.LayerNorm(self.seq_len)

        self.ConvLayers = nn.ModuleList()

        for ci in range(0, self.number_filter_types):
            self.ConvLayers.append(nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.number_filters,
                                             kernel_size=self.filter_dim[ci],
                                             padding=int((self.filter_dim[ci] - 1) / 2)))

        self.Flatten = nn.Flatten()

        self.Dropout = nn.Dropout(dropout)

        # self.SAHead = TransformerSelfAttentionHead(seq_len=self.seq_len, embed_dim=self.seq_len *
        # self.number_filter_types * self.number_filters, hidden_units=self.hidden_units, dropout=dropout).double()

        self.TransformerHeads = nn.ModuleList()

        for ti in range(0, self.sa_modules):
            self.TransformerHeads.append(TransformerHead(seq_len=self.seq_len,
                                                         embed_dim=self.seq_len * self.number_filter_types * self.number_filters,
                                                         sa_heads=self.sa_heads,
                                                         hidden_units_sa=int(self.hidden_units_sa / self.sa_heads),
                                                         hidden_units_out=self.hidden_units_out,
                                                         dropout=self.dropout))

        self.LN1 = nn.LayerNorm([self.embedding_dim, self.seq_len])
        self.LN2 = nn.LayerNorm(self.hidden_units_out)

        self.OutputLayer = nn.Linear(in_features=self.hidden_units_out,
                                     out_features=self.num_labels, bias=True)

    def forward(self, seq):

        y = self.Embedding(seq).transpose(1, 2)

        # add the positional embedding
        y = Summer(self.PositionEmbedding)(y)

        # do a normalization
        y = self.LN1(y)

        f = []
        for ci in range(0, len(self.ConvLayers)):
            f.append(self.ConvLayers[ci](y))

        fc = torch.cat(tuple(f), 1)

        fc_ = self.Flatten(fc)

        fc2 = fc_.reshape(tuple([1] + list(fc_.shape)))

        # this may have to be reevaluated not sure if we need to add sequential skip connections
        # between attention heads or simply with inputs
        ZF = self.TransformerHeads[0](fc2)

        if self.layer_norm:
            X_ = self.LN2(ZF + fc2)
        else:
            X_ = ZF + fc2

        if self.sa_modules > 1:

            for ti in range(1,self.sa_modules):
                ZF = self.TransformerHeads[ti](X_)

            if self.layer_norm:
                X_ = self.LN2(ZF + X_)
            else:
                X_ = ZF + X_

        X_ = X_.reshape(fc_.shape[0], self.hidden_units_out)

        output = torch.sigmoid(self.OutputLayer(X_))

        return output
