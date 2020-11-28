import json
import pickle
import re
from collections import Counter
from typing import Dict, List, Optional
from pathlib import Path

import nltk


DEFAULT_PADDING_TOKEN = "<pad>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_OOV_TOKEN = "<unk>"


class Vocabulary(object):
    def __init__(
        self,
        padding_token: Optional[str] = DEFAULT_PADDING_TOKEN,
        oov_token: Optional[str] = DEFAULT_OOV_TOKEN,
        bos_token: Optional[str] = DEFAULT_BOS_TOKEN,
        eos_token: Optional[str] = DEFAULT_EOS_TOKEN
    ) -> None:
        self.padding_token = padding_token \
            if padding_token is not None else DEFAULT_PADDING_TOKEN
        self.oov_token = oov_token if oov_token is not None else DEFAULT_OOV_TOKEN
        self.bos_token = bos_token if bos_token is not None else DEFAULT_BOS_TOKEN
        self.eos_token = eos_token if eos_token is not None else DEFAULT_EOS_TOKEN
        self.w2i = {
            self.padding_token: 0,
            self.oov_token: 1,
            self.bos_token: 2,
            self.eos_token: 3
        }
        self.i2w = self.generate_inverse_dict(self.w2i)

    def __len__(self) -> int:
        return len(self.w2i)

    def build(self, sentences: List[List[str]], mincount: int = 1) -> "Vocabulary":
        counter = {}
        for sentence in sentences:
            for token in sentence:
                counter[token] = counter.get(token, 0) + 1

        for token, count in sorted(counter.items(), key=lambda x: x[1], reverse=True):
            if count < mincount:
                break
            index = len(self.w2i)
            self.w2i[token] = index
            self.i2w[index] = token
        return self

    def add(self, token: str) -> None:
        index = len(self.w2i)
        val = self.w2i.setdefault(token, index)
        if val == index:
            self.i2w[index] = token

    def get_index(self, token: str) -> int:
        return self.w2i.get(token, self.w2i[self.oov_token])

    def get_token(self, index: int) -> str:
        vocab_size = len(self.i2w)
        if index >= vocab_size:
            raise IndexError(f"Vocabulary size is {vocab_size}, got {index} as index")
        return self.i2w[index]

    def get_dict(self) -> Dict[str, int]:
        return self.w2i.copy()

    def generate_inverse_dict(self, w2i: Dict[str, int]) -> Dict[int, str]:
        i2w = {}
        for k, v in w2i.items():
            i2w[v] = k
        return i2w

    def extend_from_vocab(self, vocab: "Vocabulary") -> None:
        for token in vocab.w2i.keys():
            self.add(token)

    def load(self, path: Path) -> None:
        with open(path, "r") as f:
            self.w2i = json.load(f)
        self.i2w = self.generate_inverse_dict(self.w2i)

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(self.w2i, f, ensure_ascii=False, indent=4)


def build_vocab(
    factual_caption_path: Path,
    styled_caption_path_list: List[Path]
) -> Vocabulary:
    vocab = Vocabulary()
    factual_captions = extract_factual_captions(factual_caption_path)
    vocab.build(factual_captions, mincount=2)

    for caption_path in styled_caption_path_list:
        vocab_styled = Vocabulary()
        styled_captions = extract_styled_captions(caption_path)
        vocab_styled.build(styled_captions, mincount=1)
        vocab.extend_from_vocab(vocab_styled)
    return vocab


def extract_factual_captions(
    file_path: Path = Path("data/factual_train.txt")
) -> List[List[str]]:
    with open(file_path, "r") as f:
        lines = f.readlines()

    captions = []
    pattern = re.compile(r"\d*.jpg#\d*")
    for line in lines:
        line = pattern.sub("", line)
        line = line.replace(".", "")
        line = line.strip()
        words = nltk.tokenize.word_tokenize(line)
        captions.append(words)
    return captions


def extract_styled_captions(
    file_path: Path = Path("data/humor/funny_train.txt")
) -> List[List[str]]:
    with open(file_path, "r") as f:
        lines = f.readlines()

    captions = []
    for line in lines:
        line = line.replace(".", "")
        line = line.strip()
        words = nltk.tokenize.word_tokenize(line)
        captions.append(words)
    return captions


if __name__ == "__main__":
    factual_caption_path = Path("data/factual_train.txt")
    humorous_caption_path = Path("data/humor/funny_train.txt")
    vocab_save_path = Path("data/vocab.json")
    vocab = build_vocab(factual_caption_path, humorous_caption_path)
    vocab.save(vocab_save_path)
