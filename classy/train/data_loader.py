import numpy as np
import torch
import os


def random_split_data(input_, output_, split_ratio):
    N = len(input_)
    rand_idx = np.random.permutation(range(0, N))
    tn = int(np.round(split_ratio * N))
    train_idx = rand_idx[0:tn]

    val_idx = rand_idx[tn:]

    input_train = input_[train_idx]
    output_train = output_[train_idx]

    input_val = input_[val_idx]
    output_val = output_[val_idx]

    return input_train, input_val, output_train, output_val


def LoadData(directory, split_ratio=0.7, downsample=1,
             binary_target=True, binary_threshold=0.5):
    mat_path = os.path.join(directory, 'embedding_matrix.npy')
    tkn_path = os.path.join(directory, 'tokens.npy')
    out_path = os.path.join(directory, 'labels.npy')

    M = np.load(mat_path)
    input_ = np.load(tkn_path)
    output = np.load(out_path)

    if downsample > 1:
        N = input_.shape[0]
        ind = np.arange(0, N, downsample)
        input_ = input_[ind, :]
        output = output[ind]

    if binary_target:
        output[output > binary_threshold] = 1
        output[output <= (1 - binary_threshold)] = 0

        mask = (output > binary_threshold) | (output <= (1 - binary_threshold))
        output = output[mask]
        input_ = input_[mask]

    embedding_matrix = torch.tensor(M).type(torch.float32)
    input_ = torch.tensor(input_)
    output = torch.tensor(output).type(torch.float32)

    input_t, input_test, output_t, output_test = \
        random_split_data(input_, output, split_ratio)

    input_train, input_val, output_train, output_val = \
        random_split_data(input_t, output_t, split_ratio)

    del input_, output, input_t, output_t
    import gc
    gc.collect()

    input_dic = {'train': input_train, 'val': input_val, 'test': input_test}
    output_dic = {'train': output_train, 'val': output_val, 'test': output_test}

    return input_dic, output_dic, embedding_matrix
