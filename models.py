from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch


class AttnLSTMDecoder(nn.Module):
    def __init__(self, config):
        super(AttnLSTMDecoder, self).__init__()
        self.hidden_size = config['model']['decoder']['hidden_size']
        self.output_size = config['tgt_vocab_size']
        self.num_layers = config['model']['decoder']['num_layers']
        self.dropout_p = config['model']['decoder']['dropout']
        self.bidir = config['model']['decoder']['bidirectional']
        self.embed_size = config['model']['embed_size']
        self.embedding = config['src_embedding_matrix']
        self.batch_size = config['model']['batch_size']

        self.max_length = config['max_length']


        self.attn = nn.Linear(self.hidden_size + self.embed_size, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size + self.embed_size, self.embed_size)
        self.dropout = nn.Dropout(self.dropout_p)

        self.lstm = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size, dropout=self.dropout_p,
                            num_layers=self.num_layers, bidirectional=self.bidir)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, cell, encoder_outputs):
        embedded = self.embedding(input).view(1, self.batch_size, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)

        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        output, (hidden ,cell) = self.lstm(output, (hidden, cell))

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights



class LSTMEncoder(nn.Module):
    def __init__(self, config):
        super(LSTMEncoder, self).__init__()
        self.embed_size = config['model']['embed_size']
        self.batch_size = config['model']['batch_size']
#         self.batch_size = config['model']['batch_size']

        self.hidden_size = config['model']['encoder']['hidden_size']
        self.num_layers = config['model']['encoder']['num_layers']
        self.bidir = config['model']['encoder']['bidirectional']
        if self.bidir:
            self.direction = 2
        else: self.direction = 1
        self.dropout = config['model']['encoder']['dropout']

        self.embedding = config['src_embedding_matrix']
        self.lstm = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size, dropout=self.dropout,
                            num_layers=self.num_layers, bidirectional=self.bidir)

    def initHiddenCell(self):
        rand_hidden = Variable(torch.randn(self.direction * self.num_layers, self.batch_size, self.hidden_size))
        rand_cell = Variable(torch.randn(self.direction * self.num_layers, self.batch_size, self.hidden_size))
        if torch.cuda.is_available():
            rand_hidden = rand_hidden.cuda()
            rand_cell = rand_cell.cuda()
        return rand_hidden, rand_cell

    def forward(self, input, hidden, cell):
        input = self.embedding(input).view(1,self.batch_size, -1)
        output, (hidden, cell) = self.lstm(input, (hidden, cell))
        return output, hidden, cell