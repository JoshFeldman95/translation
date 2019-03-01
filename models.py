import torch
from namedtensor import ntorch, NamedTensor
import numpy as np
import random

class LSTMEncoder(ntorch.nn.Module):
    def __init__(self, DE, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.input_dim = len(DE.vocab)
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = ntorch.nn.Embedding(self.input_dim, self.emb_dim).spec('srcSeqlen','embedding')

        self.rnn = ntorch.nn.LSTM(self.emb_dim, self.hid_dim, self.n_layers, dropout=self.dropout).spec("embedding", "srcSeqlen", "lstm")

        self.dropout = ntorch.nn.Dropout(dropout)

    def forward(self, x):
        x = x[{'srcSeqlen':slice(-1,0)}]
        x = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.rnn(x)
        return hidden, cell

class LSTMDecoder(ntorch.nn.Module):
    def __init__(self, EN, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = len(EN.vocab)
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = ntorch.nn.Embedding(self.output_dim, self.emb_dim).spec('trgSeqlen','embedding')

        self.rnn = (ntorch.nn.LSTM(self.emb_dim, self.hid_dim, self.n_layers, dropout=self.dropout)
                    .spec("embedding", "trgSeqlen", "lstm"))

        self.out = ntorch.nn.Linear(self.hid_dim, self.output_dim).spec('lstm','out')

        self.dropout = ntorch.nn.Dropout(dropout)

    def forward(self, x, hidden, cell):
        x = self.embedding(x)
        x = self.dropout(x)
        x, (hidden, cell) = self.rnn(x, (hidden, cell))
        x = self.dropout(x)
        x = self.out(x)
        return x, hidden, cell

class Seq2Seq(ntorch.nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.val_loss = float('inf')

        assert encoder.hid_dim == decoder.hid_dim, "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        trg_vocab_size = self.decoder.output_dim

        #tensor to store decoder outputs
        outputs = torch.zeros(trg.shape['trgSeqlen'], trg.shape['batch'], trg_vocab_size).to(self.device)

        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)

        #first input to the decoder is the <sos> tokens
        inpt = trg[{'trgSeqlen': slice(0,1)}]

        for t in range(1, trg.shape['trgSeqlen']):
            output, hidden, cell = self.decoder(inpt, hidden, cell)
            outputs[t] = output[{'trgSeqlen':0}].values
            top1 = output.max("out")[1]
            teacher_force = random.random() < teacher_forcing_ratio
            inpt = (trg[{'trgSeqlen':slice(t,t+1)}] if teacher_force else top1)
        outputs = NamedTensor(outputs, names = ('trgSeqlen', 'batch', 'out'))
        return outputs

    def fit(self, train_iter, val_iter=[], lr=1e-2, verbose=True,
        batch_size=128, epochs=10, interval=1, early_stopping=False):
        self.to(self.device)
        lr = torch.tensor(lr)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        train_iter.batch_size = batch_size

        for epoch in range(epochs):  # loop over the dataset multiple times
            self.train()
            running_loss = 0.0
            self.train()
            for i, data in enumerate(train_iter, 0):
                # get the inputs
                src, trg = data.src, data.trg

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                out = self(src, trg)
                loss = criterion(
                    out[{"trgSeqlen":slice(1,out.shape["trgSeqlen"])}].transpose("batch", "out", "trgSeqlen").values,
                    trg[{"trgSeqlen":slice(1,trg.shape["trgSeqlen"])}].transpose("batch", "trgSeqlen").values,
                )
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % interval == interval - 1:  # print every 2000 mini-batches
                    if verbose:
                        print(
                            "[epoch: {}, batch: {}] loss: {}".format(
                                epoch + 1, i + 1, running_loss / interval
                            )
                        )
                    running_loss = 0.0

            running_loss = 0.0
            val_count = 0.0
            self.eval()
            for i, data in enumerate(val_iter):
                src, trg = data.src, data.trg
                out = self(src, trg, teacher_forcing_ratio = 0)
                loss = criterion(
                    out[{"trgSeqlen":slice(1,out.shape["trgSeqlen"])}].transpose("batch", "out", "trgSeqlen").values,
                    trg[{"trgSeqlen":slice(1,trg.shape["trgSeqlen"])}].transpose("batch", "trgSeqlen").values,
                )
                running_loss += loss.item()
                val_count += 1
            prev_loss = self.val_loss
            self.val_loss = running_loss / val_count
            if verbose:
                print(f'Val loss: {self.val_loss}, PPL: {np.exp(self.val_loss)}')
            if self.val_loss > prev_loss and early_stopping:
                break
            lr *= .8
