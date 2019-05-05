import torch
import torch.nn as nn

class BidirectionalLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, dropout=True):

        #x = [sent len, batch size]
        if dropout:
            embedded = self.dropout(self.embedding(x))
        else:
            embedded = self.embedding(x)

        #embedded = [sent len, batch size, emb dim]

        if dropout:
            output, (hidden, cell) = self.dropout(self.rnn(embedded))
        else:
            output, (hidden, cell) = self.rnn(embedded)

        #output = [sent len, batch size, hid dim * num directions]
        #hidden = [num layers * num directions, batch size, hid dim]
        #cell = [num layers * num directions, batch size, hid dim]

        #concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        #and apply dropout
        if dropout:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        #hidden = [batch size, hid dim * num directions]

        return self.fc(hidden.squeeze(0))

    def __call__(self, x):
        return self.forward(x, dropout=False)
