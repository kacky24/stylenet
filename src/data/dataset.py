import os
import re
from typing import Callable, List, Tuple

import nltk

import numpy as np

import skimage.io
import skimage.transform

import torch
from torch.utils.data import Dataset

from src.utils.vocab import Vocab


class Flickr7kDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        caption_file: str,
        vocab: Vocab,
        transform: Callable[[np.ndarray], np.ndarray] = None
    ) -> None:
        """
        Args:
            img_dir: Direcutory with all the images
            caption_file: Path to the factual caption file
            vocab: Vocab instance
            transform: Optional transform to be applied
        """
        self.img_dir = img_dir
        self.imgname_caption_list = self._get_imgname_and_caption(caption_file)
        self.vocab = vocab
        self.transform = transform

    def _get_imgname_and_caption(self, caption_file: str) -> List[str]:
        with open(caption_file, "r") as f:
            res = f.readlines()

        imgname_caption_list = []
        r = re.compile(r"#\d*")
        for line in res:
            img_and_cap = r.split(line)
            img_and_cap = [x.strip() for x in img_and_cap]
            imgname_caption_list.append(img_and_cap)

        return imgname_caption_list

    def __len__(self) -> int:
        return len(self.imgname_caption_list)

    def __getitem__(self, ix: int) -> Tuple[torch.Tensor]:
        img_name = self.imgname_caption_list[ix][0]
        img_name = os.path.join(self.img_dir, img_name)
        caption = self.imgname_caption_list[ix][1]

        image = skimage.io.imread(img_name)
        if self.transform is not None:
            image = self.transform(image)

        # convert caption to word ids
        r = re.compile("\.")
        tokens = nltk.tokenize.word_tokenize(r.sub("", caption).lower())
        caption = []
        caption.append(self.vocab("<s>"))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab("</s>"))
        caption = torch.Tensor(caption)
        return image, caption


class FlickrStyle7kDataset(Dataset):
    def __init__(self, caption_file: str, vocab: Vocab) -> None:
        """
        Args:
            caption_file: Path to styled caption file
            vocab: Vocab instance
        """
        self.caption_list = self._get_caption(caption_file)
        self.vocab = vocab

    def _get_caption(self, caption_file: str) -> List[str]:
        with open(caption_file, "r") as f:
            caption_list = f.readlines()

        caption_list = [x.strip() for x in caption_list]
        return caption_list

    def __len__(self) -> int:
        return len(self.caption_list)

    def __getitem__(self, ix: int) -> torch.Tensor:
        caption = self.caption_list[ix]
        # convert caption to word ids
        r = re.compile("\.")
        tokens = nltk.tokenize.word_tokenize(r.sub("", caption).lower())
        caption = []
        caption.append(self.vocab("<s>"))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab("</s>"))
        caption = torch.Tensor(caption)
        return caption
