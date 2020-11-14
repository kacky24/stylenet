import pickle
from typing import Callable, List, Tuple

import numpy as np

import torch
from torch.utils.data import DataLoader

from torchvision import transforms

from src.data.dataset import Flickr7kDataset
from src.data.dataset import FlickrStyle7kDataset
from src.data.transforms import Rescale
from src.utils.vocab import Vocab


def get_data_loader(
    img_dir: str,
    caption_file: str,
    vocab: Vocab,
    batch_size: int,
    transform: Callable([np.ndarray], np.ndarray) = None,
    shuffle=False,
    num_workers=0
) -> DataLoader:
    if transform is None:
        transform = transforms.Compose([
            Rescale((224, 224)),
            transforms.ToTensor()
        ])

    flickr7k = Flickr7kDataset(img_dir, caption_file, vocab, transform)

    data_loader = DataLoader(dataset=flickr7k,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers,
                             collate_fn=collate_fn)
    return data_loader


def get_styled_data_loader(
    caption_file: str,
    vocab: Vocab,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0
) -> DataLoader:
    flickr_styled_7k = FlickrStyle7kDataset(caption_file, vocab)

    data_loader = DataLoader(dataset=flickr_styled_7k,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers,
                             collate_fn=collate_fn_styled)
    return data_loader


def collate_fn(data: List[Tuple[np.ndarray]]) -> Tuple[torch.Tensor]:
    """
    create minibatch tensors from data(list of tuple(image, caption))
    """
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # images : tuple of 3D tensor -> 4D tensor
    images = torch.stack(images, 0)

    # captions : tuple of 1D Tensor -> 2D tensor
    lengths = torch.LongTensor([len(cap) for cap in captions])
    captions = [pad_sequence(cap, max(lengths)) for cap in captions]
    captions = torch.stack(captions, 0)

    return images, captions, lengths


def collate_fn_styled(captions: List[np.ndarray]):
    captions.sort(key=lambda x: len(x), reverse=True)

    # tuple of 1D Tensor -> 2D Tensor
    lengths = torch.LongTensor([len(cap) for cap in captions])
    captions = [pad_sequence(cap, max(lengths)) for cap in captions]
    captions = torch.stack(captions, 0)

    return captions, lengths


def pad_sequence(seq: torch.Tensor, max_len: int) -> torch.Tensor:
    seq = torch.cat((seq, torch.zeros(max_len - len(seq))))
    return seq


if __name__ == "__main__":
    with open("data/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    img_path = "data/flickr7k_images"
    cap_path = "data/factual_train.txt"
    cap_path_styled = "data/humor/funny_train.txt"
    data_loader = get_data_loader(img_path, cap_path, vocab, 3)
    styled_data_loader = get_styled_data_loader(cap_path_styled, vocab, 3)

    for i, (captions, lengths) in enumerate(styled_data_loader):
        print(i)
        # print(images.shape)
        print(captions[:, 1:])
        print(lengths - 1)
        print()
        if i == 3:
            break
