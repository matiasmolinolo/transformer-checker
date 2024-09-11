import json
from typing import List

import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


class DyckLanguageTokenizer:
    START_TOKEN, PAD_TOKEN, END_TOKEN = 0, 1, 2
    base_vocab = {"[start]": START_TOKEN, "[pad]": PAD_TOKEN, "[end]": END_TOKEN}

    def __init__(self, vocab: str):
        self.vocab = vocab
        self.tok_to_i = {
            **{tok: i + 3 for i, tok in enumerate(vocab)},
            **self.base_vocab,
        }
        self.i_to_tok = {i: tok for tok, i in self.tok_to_i.items()}
        
        # Precompute tokenization mapping
        self.char_to_token = np.zeros(256, dtype=np.float32)
        for char, token in self.tok_to_i.items():
            if len(char) == 1:
                self.char_to_token[ord(char)] = token

    def tokenize(self, strings: str | List[str], max_len=None):
        if isinstance(strings, str):
            strings = [strings]

        if max_len is None:
            max_len = max((max(len(s) for s in strings)), 1)

        # Vectorized tokenization
        tokenized = np.full((len(strings), max_len + 2), self.PAD_TOKEN, dtype=np.float32)
        tokenized[:, 0] = self.START_TOKEN
        
        lengths = np.array([len(s) for s in tqdm(strings, desc="Calculating lengths")])
        for i, s in enumerate(tqdm(strings, desc="Tokenizing strings")):
            tokenized[i, 1:lengths[i]+1] = self.char_to_token[[ord(c) for c in s]]
        
        # Efficient end token placement
        tokenized[np.arange(len(strings)), lengths + 1] = self.END_TOKEN

        return torch.from_numpy(tokenized)

    def decode(self, tokens, remove_special_tokens=True):
        if tokens.ndim < 2:
            raise ValueError("Needs to have a batch dimension.")

        def i_to_c(i):
            if i < len(self.i_to_tok):
                return self.i_to_tok[i]
            raise ValueError(f"Index {i} not in vocabulary")

        if remove_special_tokens:
            return [
                "".join(i_to_c(i.item()) for i in seq[1:] if i != self.START_TOKEN and i != self.END_TOKEN)
                for seq in tokens
            ]
        return [" ".join(i_to_c(i.item()) for i in seq) for seq in tokens]

    def decode_single(self, tokens, remove_special_tokens=True):
        return self.decode(tokens.unsqueeze(0), remove_special_tokens=remove_special_tokens)[0]

    def __repr__(self):
        return f"DyckLanguageTokenizer(vocab={self.vocab!r})"


class DyckLanguageDataset(Dataset):
    def __init__(self, file: str | list, vocab: str):
        self.vocab = vocab
        self.tokenizer = DyckLanguageTokenizer(vocab)
        self.data = []
        if isinstance(file, list):
            self.data = file
        elif isinstance(file, str):
            with open(file, "r") as f:
                for line in f:
                    self.data.append(json.loads(line))
                print(f"Loaded {len(self.data)} samples from {file}")

        self.strings = [sample[0] for sample in self.data]
        self.tokenized = self.tokenizer.tokenize(self.strings)
        self.balanced = torch.tensor([sample[1] for sample in self.data], dtype=torch.float)

    def to(self, device):
        self.tokenized = self.tokenized.to(device)
        self.balanced = self.balanced.to(device)
        return self

    def __len__(self):
        return len(self.strings)

    def __getitem__(self, idx):
        if type(idx) == slice:
            return self.__class__(list(zip(self.strings[idx], self.balanced[idx])), self.vocab)

        return (self.strings[idx], self.balanced[idx], self.tokenized[idx])
