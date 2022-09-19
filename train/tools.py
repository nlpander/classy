import time
import torch
import numpy as np

from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

global device
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


def BatchEvaluation(model, criterion, input_val, output_val, batch_size, mode='classification', multi_label=False,
                    precision='torch.float32'):
    model.eval()
    val_steps = [k for k in range(0, input_val.shape[0], batch_size)]

    f1_val = []
    loss_val = []

    for j in range(1, len(val_steps)):

        input_val_batch = input_val[val_steps[j - 1]:val_steps[j], :]
        output_val_batch = output_val[val_steps[j - 1]:val_steps[j]]

        input_val_batch = input_val_batch.to(device=device, dtype=torch.int32)
        output_val_batch = output_val_batch.to(device=device, dtype=eval(precision))

        y_val_out = model(input_val_batch)

        if multi_label:
            loss = criterion(y_val_out, output_val_batch)
        else:
            loss = criterion(y_val_out.view(y_val_out.shape[0]), output_val_batch)

        loss_val_b = loss.item()
        loss_val.append(loss_val_b)

        output_val_batch = output_val_batch.to('cpu').detach().numpy()
        y_val_out = y_val_out.to('cpu').detach().numpy()

        f1_val_b = f1_score(y_val_out.round(), output_val_batch.round(), average="weighted")
        f1_val.append(f1_val_b)

    loss_val_ = np.mean(loss_val)
    f1_val_ = np.mean(f1_val)

    return f1_val_, loss_val_


class TrainTestNetworkES:

    def __init__(self, model, input_, output_, nb_epochs, batch_size, criterion, optimizer, f1_threshold, mse_threshold,
                 batch_progress=True, mode='classification', multi_label=False, precision='torch.float32'):

        self.model = model
        self.input_train = input_['train']
        self.input_val = input_['val']
        self.input_test = input_['test']

        self.output_train = output_['train']
        self.output_val = output_['val']
        self.output_test = output_['test']

        self.nb_epochs = nb_epochs
        self.actual_epochs = 0
        self.batch_size = batch_size
        self.criterion = criterion
        self.optimizer = optimizer
        self.f1_threshold = f1_threshold
        self.mse_threshold = mse_threshold
        self.batch_progress = batch_progress
        self.mode = mode
        self.multi_label = multi_label
        self.precision = precision

        self.epoch_f1_val = None
        self.epoch_loss_val = None
        self.epoch_f1_train = None
        self.epoch_loss_train = None
        self.f1_test = None
        self.loss_test = None

    def train(self):

        steps = [k for k in range(0, self.input_train.shape[0], self.batch_size)]

        ep_loss = []
        ep_loss_val = []

        ep_f1_train = []
        ep_f1_val = []

        t0 = time.time()

        # intialize the epoch
        n = 0

        # initialize the fl values
        f1_train = 0
        f1_val = 0

        # intialize the loss values
        epoch_loss = 1
        loss_val = 1

        if self.mode == 'classification':
            criteria = (f1_train < self.f1_threshold or f1_val < self.f1_threshold)
        elif self.mode == 'regression':
            criteria = (epoch_loss > self.mse_threshold or loss_val > self.mse_threshold)

        while criteria and n < self.nb_epochs:

            epoch_losses = []
            epoch_f1 = []

            for i in range(1, len(steps)):

                self.model.train()
                self.optimizer.zero_grad()

                input_batch = self.input_train[steps[i - 1]:steps[i], :]
                output_batch = self.output_train[steps[i - 1]:steps[i]]

                input_batch = input_batch.to(device=device, dtype=torch.int32)
                output_batch = output_batch.to(device=device, dtype=eval(self.precision))

                y_out = self.model(input_batch)

                if self.multi_label:
                    loss = self.criterion(y_out, output_batch)
                else:
                    loss = self.criterion(y_out.view(y_out.shape[0]), output_batch)

                loss.backward()
                self.optimizer.step()

                epoch_losses.append(loss.item())
                y_out_ = y_out.to('cpu').detach().numpy()

                input_batch_ = input_batch.to('cpu').detach().numpy()
                output_batch_ = output_batch.to('cpu').detach().numpy()

                f1 = f1_score(y_out_.round(), output_batch_.round(), average='weighted')
                epoch_f1.append(f1)

                if self.batch_progress:
                    print('batch %d of %d complete' % (i, len(steps)))

            epoch_loss = np.mean(epoch_losses)
            ep_loss.append(epoch_loss)

            f1_train = np.mean(epoch_f1)
            ep_f1_train.append(f1_train)

            # print the epoch training Loss
            print('')
            print("Epoch: %d, training loss: %.4f, training f1 score: %.4f" % (n + 1, epoch_loss, f1_train))

            # evaluate on the validation set
            with torch.no_grad():
                f1_val, loss_val = BatchEvaluation(self.model, self.criterion,
                                                   self.input_val, self.output_val,
                                                   self.batch_size, mode=self.mode,
                                                   multi_label=self.multi_label,
                                                   precision=self.precision)

                ep_loss_val.append(loss_val)
                ep_f1_val.append(f1_val)

                # print the epoch validation loss
                print('Epoch: %d, validation loss: %.4f, validation f1 score: %.4f' % (n + 1, loss_val, f1_val))
                print('############################################################################')
                print('')
                n += 1

            # upgrade criteria for early stopping
            if self.mode == 'classification':
                criteria = (f1_train < self.f1_threshold or f1_val < self.f1_threshold)
            elif self.mode == 'regression':
                criteria = (epoch_loss > self.mse_threshold or loss_val > self.mse_threshold)

        self.actual_epochs = n
        self.epoch_loss_train = ep_loss
        self.epoch_f1_train = ep_f1_train
        self.epoch_loss_val = ep_loss_val
        self.epoch_f1_val = ep_f1_val

        print('%.2f seconds have elapsed..' % (time.time() - t0))

    def test(self):

        with torch.no_grad():
            f1_test, loss_test = BatchEvaluation(self.model, self.criterion,
                                                 self.input_test, self.output_test,
                                                 self.batch_size, mode=self.mode,
                                                 multi_label=self.multi_label,
                                                 precision=self.precision)

        self.loss_test = loss_test
        self.f1_test = f1_test

        print("Test loss: %.4f, Test fl score: %.4f" % (self.loss_test, self.f1_test))

    def plot_training_loss(self):

        # ### plot training and validation error
        f = plt.figure(figsize=(12, 4))
        x = range(0, self.actual_epochs)

        if self.mode == 'classification':

            ax = f.add_subplot(122)
            ax2 = f.add_subplot(121)

            ax.plot(x, self.epoch_f1_train, x, self.epoch_f1_val)
            ax.set_xlabel('number of epochs')
            ax.set_ylabel('f1-score')

            ax.legend(['train', 'validation'])

            ax2.plot(x, self.epoch_loss_train, x, self.epoch_loss_val)
            ax2.set_xlabel('number of epochs')
            ax2.set_ylabel('log-loss')
            ax2.legend(['train', 'validation'])

        elif self.mode == 'regression':

            plt.plot(x, self.epoch_loss_train, x, self.epoch_loss_val)
            plt.xlabel('number of epochs')

            plt.ylabel('mean squared error')
            plt.legend(['train', 'validation'])