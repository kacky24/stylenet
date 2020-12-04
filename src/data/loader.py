from pathlib import Path
from typing import Callable, List, Tuple, Optional

import numpy as np

import torch
from torch.utils.data import DataLoader

from torchvision import transforms

from src.data.dataset import Flickr7kDataset
from src.data.dataset import FlickrStyle7kDataset
from src.data.transforms import Rescale
from src.utils.vocab import Vocabulary


def get_factual_data_loader(
    img_dir: Path,
    caption_path: Path,
    vocab: Vocabulary,
    batch_size: int,
    transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    shuffle: bool = False,
    num_workers: int = 0
) -> DataLoader:
    if transform is None:
        transform = transforms.Compose([
            Rescale((224, 224)),
            transforms.ToTensor()
        ])

    flickr7k = Flickr7kDataset(img_dir, caption_path, vocab, transform)

    data_loader = DataLoader(dataset=flickr7k,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers,
                             collate_fn=collate_fn_factual)
    return data_loader


def get_styled_data_loader(
    caption_path: Path,
    vocab: Vocabulary,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0
) -> DataLoader:
    flickr_styled_7k = FlickrStyle7kDataset(caption_path, vocab)

    data_loader = DataLoader(dataset=flickr_styled_7k,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers,
                             collate_fn=collate_fn_styled)
    return data_loader


def collate_fn_factual(data: List[Tuple[np.ndarray]]) -> Tuple[torch.Tensor]:
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
    seq = torch.cat((seq, torch.zeros(max_len - len(seq), dtype=torch.long)))
    return seq
