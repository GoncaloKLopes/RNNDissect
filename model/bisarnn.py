import torch
import torch.nn as nn


class BinarySARNN(nn.Module):

    def __init__(self, config):
        super(BinarySARNN, self).__init__()
        self.hidden_dim = config.d_hidden
        self.vocab_size = config.vocab_size
        self.embed_dim = config.d_embed
        self.batch_size = config.batch_size
        self.num_labels = 2
        self.num_layers = config.n_layers

        self.dropout = config.dropout
        self.bidir = config.bidir
        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.arch = config.arch
        if config.arch == "RNN":
            if config.nonlin not in ("tanh", "relu"):
                raise ValueError("Invalid activation function", config.nonlin,
                                 ". Expected \"tanh\" or \"relu\".")
            else:
                self.nonlin = config.nonlin

            self.rnn = nn.RNN(input_size=self.embed_dim,
                              hidden_size=self.hidden_dim,
                              num_layers=self.num_layers,
                              dropout=self.dropout,
                              nonlinearity=self.nonlin,
                              bidirectional=self.bidir)
        elif config.arch == "LSTM":
            self.rnn = nn.LSTM(input_size=self.embed_dim,
                               hidden_size=self.hidden_dim,
                               num_layers=self.num_layers,
                               dropout=self.dropout,
                               bidirectional=self.bidir)
        else:
            raise ValueError("Invalid architecture string" + config.arch)
        self.hidden_to_label = nn.Linear(self.hidden_dim * (1 + self.bidir), self.num_labels)

    def forward(self, batch, text_lengths):

        embeddings = self.embed(batch)
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(embeddings, text_lengths)

        if self.arch == "RNN":
            packed_rnn_out, hidden = self.rnn(packed_embeddings)
        elif self.arch == "LSTM":
            packed_rnn_out, (hidden, cell) = self.rnn(packed_embeddings)
        if self.bidir:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1).squeeze(0)

        else:
            if self.num_layers > 1:
                hidden = hidden[-1]
            else:
                hidden = hidden.squeeze(0)
        return self.hidden_to_label(hidden)
    
    def forward(self, sentence):
        embeddings = self.embed(sentence)

        if self.arch == "RNN":
            packed_rnn_out, hidden = self.rnn(embeddings)
        elif self.arch == "LSTM":
            packed_rnn_out, (hidden, cell) = self.rnn(embeddings)
        if self.bidir:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1).squeeze(0)

        else:
            if self.num_layers > 1:
                hidden = hidden[-1]
            else:
                hidden = hidden.squeeze(0)
        return self.hidden_to_label(hidden)
