from pathlib import Path

import pytest

import torch

from src.data.loader import get_factual_data_loader, get_styled_data_loader
from src.utils.vocab import build_vocab


@pytest.fixture(scope="module")
def vocab():
    factual_caption_path = Path("./tests/fixtures/sample_factual_data.txt")
    styled_caption_path = Path("./tests/fixtures/sample_styled_data.txt")
    vocab = build_vocab(factual_caption_path, styled_caption_path)
    return vocab


def test_get_factual_data_loader(vocab):
    img_dir = Path("./tests/fixtures/images/")
    factual_caption_path = Path("./tests/fixtures/sample_factual_data.txt")
    batch_size = 2

    factual_data_loader = get_factual_data_loader(
        img_dir,
        factual_caption_path,
        vocab=vocab,
        batch_size=batch_size,
        shuffle=False
    )
    images, captions, lengths = list(factual_data_loader)[0]
    assert images.size() == torch.Size([2, 3, 224, 224])
    assert (captions == torch.Tensor([[2, 6, 4, 5, 7, 3], [2, 6, 4, 5, 1, 3]])).all()
    assert (lengths == torch.Tensor([6, 6])).all()


def test_get_styled_data_loader(vocab):
    styled_caption_path = Path("./tests/fixtures/sample_styled_data.txt")
    batch_size = 2

    styled_data_loader = get_styled_data_loader(
        styled_caption_path,
        vocab,
        batch_size=batch_size,
        shuffle=False
    )
    captions, lengths = list(styled_data_loader)[0]
    assert (captions == torch.Tensor([[2, 6, 4, 5, 8, 3], [2, 6, 4, 5, 9, 3]])).all()
    assert (lengths == torch.Tensor([6, 6])).all()
