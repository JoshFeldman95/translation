import torch
from namedtensor import ntorch, NamedTensor
import numpy as np
import random

class LSTMEncoder(ntorch.nn.Module):
    def __init__(self, DE, emb_dim, hid_dim, n_layers, dropout, attention):
        super().__init__()
        self.attention = attention
        self.embedding = ntorch.nn.Embedding(len(DE.vocab), emb_dim).spec('srcSeqlen','embedding')
        self.rnn = ntorch.nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, bidirectional = self.attention).spec("embedding", "srcSeqlen", "lstm")
        self.dropout = ntorch.nn.Dropout(dropout)

    def forward(self, src):
        if not self.attention:
            # reverse input
            src = src[{'srcSeqlen':slice(-1,0)}]

        # run net
        x = self.embedding(src)
        x = self.dropout(x)
        outputs, hidden = self.rnn(x)
        if self.attention:
            return {'src':outputs}
        else:
            return hidden

class LSTMDecoder(ntorch.nn.Module):
    def __init__(self, EN, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = ntorch.nn.Embedding(len(EN.vocab), emb_dim).spec('trgSeqlen','embedding')
        self.rnn = ntorch.nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout).spec("embedding", "trgSeqlen", "lstm")
        self.out = ntorch.nn.Linear(hid_dim, len(EN.vocab)).spec("lstm", "logit")
        self.dropout = ntorch.nn.Dropout(dropout)

    def forward(self, trg, hidden):
        x = self.embedding(trg)
        x = self.dropout(x)
        x, hidden = self.rnn(x, hidden)
        x = self.out(x)
        return x, hidden

class AttentionDecoder(ntorch.nn.Module):
    def __init__(self, EN, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = ntorch.nn.Embedding(len(EN.vocab), emb_dim).spec('trgSeqlen','embedding')
        self.rnn = ntorch.nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout).spec("embedding", "trgSeqlen", "lstm")
        self.out = ntorch.nn.Linear(hid_dim*2, len(EN.vocab)).spec("lstm", "logit")
        self.dropout = ntorch.nn.Dropout(dropout)

    def forward(self, trg, hidden):
        # get hidden state
        src = hidden['src']
        rnn_state = hidden['rnn_state'] if 'rnn_state' in hidden else None

        #run net
        x = self.embedding(trg)
        x = self.dropout(x)
        if rnn_state is not None:
            x, rnn_state = self.rnn(x, rnn_state)
        else:
            x, rnn_state = self.rnn(x)
        context = x.dot('lstm', src).softmax('srcSeqlen').dot('srcSeqlen',src)
        x = self.out(ntorch.cat([context, x], dim = 'lstm'))

        # create new hidden state
        hidden = {'src': src, 'rnn_state':rnn_state}
        return x, hidden

class Translator(ntorch.nn.Module):
    def __init__(self, teacher_forcing, device):
        super().__init__()
        self.teacher_forcing = teacher_forcing
        self.device = device

    def forward(self, src, trg):
        #get src encoding
        hidden = self.encoder(src)

        # initialize outputs
        output_tokens = [trg[{'trgSeqlen':slice(0,1)}]]
        output_distributions = []

        # make predictions
        for t in range(trg.shape['trgSeqlen']-1):
            #predict next word
            if random.random() < self.teacher_forcing:
                inp = trg[{'trgSeqlen':slice(t,t+1)}]
                out, hidden = self.decoder(inp, hidden)
            else:
                out, hidden = self.decoder(output_tokens[t], hidden)

            #store output
            output_distributions.append(out)
            _, top1 = out.max("logit")
            output_tokens.append(top1)

        #format predictions
        return ntorch.cat(output_distributions, dim = 'trgSeqlen')

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
                src, trg = data.src, data.trg
                optimizer.zero_grad()
                out = self(src, trg)
                loss = criterion(
                    out.transpose("batch", "logit", "trgSeqlen").values,
                    trg[{"trgSeqlen":slice(1,trg.shape["trgSeqlen"])}].transpose("batch", "trgSeqlen").values,
                )
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                # print statistics
                if i % interval == interval - 1:  # print every 2000 mini-batches
                    if verbose:
                        print(f"[epoch: {epoch + 1}, batch: {i + 1}] loss: {running_loss / interval}")
                    running_loss = 0.0

            running_loss = 0.0
            val_count = 0.0
            self.eval()
            for i, data in enumerate(val_iter):
                src, trg = data.src, data.trg
                out = self(src, trg, teacher_forcing_ratio = 0)
                loss = criterion(
                    out.transpose("batch", "logit", "trgSeqlen").values,
                    trg[{"trgSeqlen":slice(1,trg.shape["trgSeqlen"])}].transpose("batch", "trgSeqlen").values
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

class LSTMTranslator(Translator):
    def __init__(self, DE, EN, src_emb_dim, trg_emb_dim, hid_dim, n_layers = 4, dropout = 0.5, teacher_forcing = 0.75, device = 'cpu'):
        super().__init__(teacher_forcing, device)
        self.encoder = LSTMEncoder(DE, src_emb_dim, hid_dim, n_layers, dropout, False)
        self.decoder = LSTMDecoder(EN, trg_emb_dim, hid_dim, n_layers, dropout)

class AttentionTranslator(Translator):
    def __init__(self, DE, EN, src_emb_dim, trg_emb_dim, hid_dim, n_layers = 4, dropout = 0.5, teacher_forcing = 0.75, device = 'cpu'):
        super().__init__(teacher_forcing, device)
        self.encoder = LSTMEncoder(DE, src_emb_dim, hid_dim, n_layers, dropout, True)
        self.decoder = AttentionDecoder(EN, trg_emb_dim, hid_dim*2, n_layers, dropout)
