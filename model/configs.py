import torch.nn as nn
import torch.optim as optim


class ModelConfig:
    """Encapsulates model configuration."""
    def __init__(self, id, d_hidden, vocab_size, d_embed,
                 batch_size, n_layers, dropout, bidir,
                 arch, nonlin=""):
        self.id = id
        self.d_hidden = d_hidden
        self.vocab_size = vocab_size
        self.d_embed = d_embed
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.nonlin = nonlin
        self.dropout = dropout
        self.bidir = bidir
        self.arch = arch

    def __str__(self):
        return "Model Config\n" + 12 * ("=") + "\n" +\
               "d_hidden = " + str(self.d_hidden) + "\n" +\
               "vocab_size = " + str(self.vocab_size) + "\n" +\
               "d_embed = " + str(self.d_embed) + "\n" +\
               "batch_size = " + str(self.batch_size) + "\n" +\
               "n_layers = " + str(self.n_layers) + "\n" +\
               "nonlin = " + str(self.nonlin) + "\n" +\
               "dropout = " + str(self.dropout) + "\n" +\
               "bidir = " + str(self.bidir) + "\n" +\
               "arch = " + self.arch


RNN_CONFIG1 = ModelConfig(id="simple_rnn", d_hidden=64, vocab_size=168994,
                          d_embed=300, batch_size=64, n_layers=1,
                          nonlin="tanh", dropout=0, bidir=False, arch="RNN")

RNN_CONFIG2 = ModelConfig(id="simple_rnn_2layers", d_hidden=128,
                          vocab_size=168994, d_embed=300,
                          batch_size=128, n_layers=2, nonlin="tanh",
                          dropout=0, bidir=False, arch="RNN")

RNN_CONFIG3 = ModelConfig(id="simple_rnn_2layers_bidir", d_hidden=128,
                          vocab_size=168994, d_embed=300,
                          batch_size=64, n_layers=2, nonlin="tanh",
                          dropout=0, bidir=True, arch="RNN")

LSTM_CONFIG1 = ModelConfig(id="lstm", d_hidden=64, vocab_size=168994,
                           d_embed=300, batch_size=64, n_layers=1,
                           dropout=0, bidir=False, arch="LSTM")

LSTM_CONFIG2 = ModelConfig(id="lstm_2layers", d_hidden=128,
                           vocab_size=168994, d_embed=300,
                           batch_size=128, n_layers=2,
                           dropout=0, bidir=False, arch="LSTM")

LSTM_CONFIG3 = ModelConfig(id="lstm_2layers_bidir", d_hidden=128,
                           vocab_size=168994, d_embed=300,
                           batch_size=64, n_layers=2,
                           dropout=0, bidir=True, arch="LSTM")
