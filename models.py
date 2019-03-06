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
        attn = x.dot('lstm', src).softmax('srcSeqlen')
        context = attn.dot('srcSeqlen',src)
        x = self.out(ntorch.cat([context, x], dim = 'lstm'))

        # create new hidden state
        hidden = {'src': src, 'rnn_state':rnn_state, 'attn': attn}
        return x, hidden


class HypothesisMap:
    def __init__(self, keys=None, vals=None, device='cpu'):
        self.device = device
        self.keys = keys or []
        self.vals = vals or []
        assert len(self.keys) == len(self.vals), (f'keys ({len(self.keys)}) and '
                                f'values ({len(self.vals)}) must have same length')
        self.key2val = {}
        for i in range(len(self.keys)):
            for b in range(self.keys[i].shape['batch']):
                self.key2val[str(self.keys[i][{'batch':b}])] = self.vals[i][{'batch':b}]

    def __contains__(self, key):
        for b in range(key.shape['batch']):
            if str(key[{'batch': b}]) not in self.key2val:
                return False
        return True

    def __getitem__(self, key):
        ret = []
        for b in range(key.shape['batch']):
            ret.append(self.key2val[str(key[{'batch': b}])])
        return ntorch.stack(ret, 'batch').to(self.device)

    def __setitem__(self, key, val):
        if str(key) in self.key2val:
            raise ValueError(f'key already in map')
        self.keys.append(key)
        self.vals.append(val)
        for i in range(len(self.keys)):
            for b in range(self.keys[i].shape['batch']):
                self.key2val[str(self.keys[i][{'batch':b}])] = self.vals[i][{'batch':b}]

    def __len__(self):
        return len(self.keys)

    def items_iter(self):
        for i in range(len(self)):
            yield self.keys[i].transpose('batch', 'trgSeqlen').to(self.device), self.vals[i].to(self.device)

    def items(self):
        return self.items_iter()

    def get_topk(self, k):
        """ get the topk items as a HypothesisMap """
        #keys = ntorch.stack(self.keys, 'map').to(self.device)
        vals = ntorch.stack(self.vals, 'map').to(self.device)
        vals, inds = vals.topk('map', k)
        keys_list = []
        for m in range(inds.shape['map']):
            newbatch = []
            for b in range(inds.shape['batch']):
                newbatch.append(self.keys[m][{'batch': b}])
            keys_list.append(ntorch.stack(newbatch, 'batch'))
        vals_list = [vals[{'map':i}] for i in range(vals.shape['map'])]
        return HypothesisMap(keys=keys_list, vals=vals_list, device=self.device)

        # keys = ntorch.tensor(keys.transpose('map', 'batch', 'trgSeqlen').values\
        #              .gather(0, inds.values.unsqueeze(-1).repeat(1, 1, keys.shape['trgSeqlen'])),
        #              names=('map', 'batch', 'trgSeqlen')).to(self.device)
        # keys_list = [keys[{'map':i}] for i in range(keys.shape['map'])]
        # vals_list = [vals[{'map':i}] for i in range(vals.shape['map'])]
        # return HypothesisMap(keys=keys_list, vals=vals_list, device=self.device)


class Translator(ntorch.nn.Module):
    def __init__(self, teacher_forcing, device):
        super().__init__()
        self.teacher_forcing = teacher_forcing
        self.device = device
        self.val_loss = float('inf')
        self.to(device)

    def beam(self, src, trg, k, beam_len, num_candidates):
        batch_size = src.shape['batch']
        out_dists = HypothesisMap(device=self.device) # map a hypothesis to distribution over words
        scores = HypothesisMap(keys=[trg[{'trgSeqlen':slice(0,1)}]],
                               vals=[ntorch.zeros(batch_size, names='batch')],
                               device=self.device) # map a hypothesis to its score
        end = HypothesisMap(device=self.device) # special buffer for hyptothesis with <EOS>
        attn = []
        EOS_IND = 3

        hidden = self.encoder(src)

        # make predictions
        for l in range(beam_len or trg.shape['trgSeqlen'] - 1):
            new_scores = HypothesisMap(device=self.device)
            hyps = scores.get_topk(k) if l > 0 else scores
            for hyp, score in hyps.items():
                inp = hyp[{'trgSeqlen':slice(l,l+1)}]

                out, hidden = self.decoder(inp, hidden)
                out = out.log_softmax('logit')
                topk = out.topk('logit', k)

                for i in range(k):
                    pred_prob = topk[0][{'logit': i, 'trgSeqlen':-1}]
                    pred = topk[1][{'logit': i}]
                    new_hyp = ntorch.cat([hyp, pred], 'trgSeqlen')

                    if hyp in out_dists:
                        out_dists[new_hyp] = ntorch.cat([out_dists[hyp], out], 'trgSeqlen')
                    else:
                        out_dists[new_hyp] = out

                    if torch.any((pred[{'trgSeqlen':-1}] == EOS_IND).values):
                        end[new_hyp] = score + pred_prob
                        end[new_hyp].masked_fill_(pred[{'trgSeqlen':-1}] != EOS_IND, -float('inf'))
                        pred_prob.masked_fill_(pred[{'trgSeqlen':-1}] == EOS_IND, -float('inf'))
                    new_scores[new_hyp] = score + pred_prob
            scores = new_scores
        for hyp, score in end.items():
            scores[hyp] = score
        best = scores.get_topk(num_candidates).keys
        out = [out_dists[k] for k in best]

        #store output
        if 'attn' in hidden:
            attn.append(hidden['attn'])

        #format predictions
        return ntorch.stack(out, 'candidates'), ntorch.cat(attn, dim = 'trgSeqlen')

    def forward(self, src, trg, teacher_forcing=None, beam_width=1, beam_len=3, num_candidates=1):
        if beam_width > 1:
            return self.beam(src, trg, beam_width, beam_len, num_candidates)
        if teacher_forcing is None:
            teacher_forcing = self.teacher_forcing
        #get src encoding
        hidden = self.encoder(src)

        # initialize outputs
        output_tokens = [trg[{'trgSeqlen':slice(0,1)}]]
        output_distributions = []
        attn = []
        # make predictions
        for t in range(trg.shape['trgSeqlen']-1):
            #predict next word
            if random.random() < teacher_forcing:
                inp = trg[{'trgSeqlen':slice(t,t+1)}]
                out, hidden = self.decoder(inp, hidden)
            else:
                out, hidden = self.decoder(output_tokens[t], hidden)
            out = out.log_softmax('logit')

            #store output
            if 'attn' in hidden:
                attn.append(hidden['attn'])
            output_distributions.append(out)
            _, top1 = out.max("logit")
            output_tokens.append(top1)

        #format predictions
        return (ntorch.cat(output_distributions, dim = 'trgSeqlen'), ntorch.cat(attn, dim = 'trgSeqlen'))
    
    def nll_loss(self, pred, target, pad_token=1):
        """ Computes the correct NLL without counting padding tokens """
        criterion = torch.nn.NLLLoss(reduction='none')
        pred = pred.transpose('batch', 'logit', 'trgSeqlen').values
        target = target[{'trgSeqlen':slice(1,pred.shape[-1]+1)}].transpose('batch', 'trgSeqlen').values
        real_tokens = (target != pad_token).to(torch.float)
        loss = criterion(pred, target)
        loss *= real_tokens
        return loss.sum() / real_tokens.sum()

    def evaluate(self, iter, beam_width=1, beam_len=3):
        self.eval()
        running_loss = 0.0
        val_count = 0.0
        for i, data in enumerate(iter):
            src, trg = data.src, data.trg
            out, _ = self.forward(src, trg, teacher_forcing=0.,
                                  beam_width=beam_width, beam_len=beam_len)
            if beam_width > 1:
                out = out[{'candidates':0}]
            loss = self.nll_loss(out, trg)
            running_loss += loss.item()
            val_count += 1
        return running_loss / val_count

    def fit(self, train_iter, val_iter=[], lr=1e-2, verbose=True,
            batch_size=128, epochs=10, interval=1, early_stopping=False):
        self.to(self.device)
        lr = torch.tensor(lr)
        criterion = torch.nn.NLLLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        train_iter.batch_size = batch_size

        for epoch in range(epochs):  # loop over the dataset multiple times
            self.train()
            running_loss = 0.0
            self.train()
            for i, data in enumerate(train_iter, 0):
                src, trg = data.src, data.trg
                optimizer.zero_grad()
                out, _ = self(src, trg)
                loss = self.nll_loss(out, trg)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                # print statistics
                if i % interval == interval - 1:  # print every 2000 mini-batches
                    if verbose:
                        print(f"[epoch: {epoch + 1}, batch: {i + 1}] loss: {running_loss / interval}")
                    running_loss = 0.0

            prev_loss = self.val_loss
            self.val_loss = self.evaluate(val_iter)
            if verbose:
                print(f'Val loss: {self.val_loss}, PPL: {np.exp(self.val_loss)}')
            if self.val_loss > prev_loss and early_stopping:
                break
            lr *= .8


class LSTMTranslator(Translator):
    def __init__(self, DE, EN, src_emb_dim, trg_emb_dim, hid_dim, n_layers = 4,
                 dropout = 0.5, teacher_forcing = 0.75, device = 'cpu', **kwargs):
        super().__init__(teacher_forcing, device, **kwargs)
        self.encoder = LSTMEncoder(DE, src_emb_dim, hid_dim, n_layers, dropout, False)
        self.decoder = LSTMDecoder(EN, trg_emb_dim, hid_dim, n_layers, dropout)


class AttentionTranslator(Translator):
    def __init__(self, DE, EN, src_emb_dim, trg_emb_dim, hid_dim, n_layers = 4,
                 dropout = 0.5, teacher_forcing = 0.75, device = 'cpu', **kwargs):
        super().__init__(teacher_forcing, device, **kwargs)
        self.encoder = LSTMEncoder(DE, src_emb_dim, hid_dim, n_layers, dropout, True)
        self.decoder = AttentionDecoder(EN, trg_emb_dim, hid_dim*2, n_layers, dropout)
