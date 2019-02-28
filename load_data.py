import torch
from torchtext import data, datasets
from namedtensor import ntorch, NamedTensor
from namedtensor.text import NamedField
import spacy

class DataLoader(object):
    def __init__(self, device, BOS_WORD = '<s>', EOS_WORD = '</s>', min_freq = 5, batch_size = 32, max_len = 20):
        self.device = torch.device(device)
        self.max_len = max_len
        self.min_freq = min_freq
        self.batch_size = batch_size
        self.BOS_WORD = BOS_WORD
        self.EOS_WORD = EOS_WORD

        self.tokenize_de, self.tokenize_en  = self._load_languages()
        self.DE, self.EN = self._create_namedfields()

    def get_iters(self):
        print("Loading data...")
        train, val, test = datasets.IWSLT.splits(
            exts=('.de', '.en'),
            fields=(self.DE, self.EN),
            filter_pred=lambda x: len(vars(x)['src']) <= self.max_len and len(vars(x)['trg']) <= self.max_len
        )

        print("building vocab...")
        self._build_vocab(train)

        print("initializing iterators...")
        train_iter, val_iter = self._get_iterators(train, val)
        return train_iter, val_iter, self.DE, self.EN

    def _load_languages(self):
        spacy_de = spacy.load('de')
        spacy_en = spacy.load('en')
        tokenize_de = self._tokenizer(spacy_de)
        tokenize_en = self._tokenizer(spacy_en)
        return tokenize_de, tokenize_en

    def _tokenizer(self, spacy_lang):
        def tokenize(text):
            return [tok.text for tok in spacy_lang.tokenizer(text)]
        return tokenize

    def _create_namedfields(self):
        DE = NamedField(names=('srcSeqlen',), tokenize=self.tokenize_de)
        EN = NamedField(names=('trgSeqlen',), tokenize=self.tokenize_en,
                        init_token = self.BOS_WORD, eos_token = self.EOS_WORD) # only target needs BOS/EOS
        return DE, EN

    def _load_data(self):
        train, val, test = datasets.IWSLT.splits(
            exts=('.de', '.en'),
            fields=(self.DE, self.EN),
            filter_pred=lambda x: (len(vars(x)['src']) <= self.max_len) and (len(vars(x)['trg']) <= self.max_len)
        )

    def _build_vocab(self, train):
        self.DE.build_vocab(train.src, min_freq=self.min_freq)
        self.EN.build_vocab(train.trg, min_freq=self.min_freq)

    def _get_iterators(self, train, val):
        train_iter, val_iter = data.BucketIterator.splits(
            (train, val),
            batch_size=self.batch_size,
            device=self.device,
            repeat=False, sort_key=lambda x: len(x.src)
        )
        return train_iter, val_iter
