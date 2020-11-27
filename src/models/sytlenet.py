from typing import List, Optional

import torch
import torch.nn as nn

from src.models.decoder import FactoredLSTM
from src.models.encoder import EncoderCNN
from src.training.loss import masked_cross_entropy


class StyleNet(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        hidden_dim: int,
        factored_dim: int,
        vocab_size: int,
        mode_list: List[str]
    ) -> None:
        super(StyleNet, self).__init__()
        self.encoder = EncoderCNN(emb_dim)
        self.decoder = FactoredLSTM(
            emb_dim, hidden_dim, factored_dim, vocab_size, mode_list)
        self.mode_list = mode_list

    def forward(
        self,
        captions: torch.Tensor,
        lengths: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        mode: str = "factual"
    ) -> torch.Tensor:
        assert bool(images) is not bool(mode == "factual")
        assert mode in self.mode_list
        if mode == "factual":
            outputs = self.forward_factual(captions, images)
            loss = masked_cross_entropy(
                outputs[:, 1:, :].contiguous(),
                captions[:, 1:].contiguous(),
                lengths - 1
            )
        else:
            outputs = self.forward_styled(captions)
            loss = masked_cross_entropy(
                outputs,
                captions[:, 1:].contiguous(),
                lengths - 1
            )
        return loss

    def forward_factual(
        self,
        captions: torch.Tensor,
        images: torch.Tensor
    ) -> torch.Tensor:
        image_features = self.encoder(images)
        outputs = self.decoder(captions, image_features)
        return outputs

    def forward_styled(self, captions: torch.Tensor) -> torch.Tensor:
        outputs = self.decoder(captions)
        return outputs

    def beam_search(
        self,
        feature: torch.Tensor,
        beam_size: int,
        max_len: int,
        mode: str = "factual"
    ) -> torch.Tensor:
        return self.decoder.beam_search(feature, beam_size, max_len, mode)
