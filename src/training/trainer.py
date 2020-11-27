from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from src.models.sytlenet import StyleNet


class Trainer(object):
    def __init__(
        self,
        model: StyleNet,
        optimizer_map: Dict[str, torch.optim.Optimizer],
        data_loader_map: Dict[str, DataLoader],
        mode_list: List[str],
        num_epochs: int,
        log_steps: int,
        save_dir: Path
    ) -> None:
        self.model = model
        self.optimizer_map = optimizer_map
        self.data_loader_map = data_loader_map
        self.mode_list = mode_list
        self.num_epochs = num_epochs
        self.log_steps = log_steps
        self.save_dir = save_dir

    def fit(self):
        for epoch in range(self.num_epochs):
            for mode in self.mode_list:
                if mode == "factual":
                    self.fit_factual_epoch(epoch, mode=mode)
                else:
                    self.fit_styled_epoch(epoch, mode=mode)
            self.save(epoch)

    def fit_factual_epoch(self, epoch: int, mode: str = "factual") -> None:
        data_loader = self.data_loader_map[mode]
        optimizer = self.optimizer_map[mode]
        total_steps = len(data_loader)
        for i, (images, captions, lengths) in enumerate(data_loader):
            self.model.zero_grad()
            loss = self.model(captions, lengths, images, mode=mode)
            loss.backward()
            optimizer.step()

            if i % self.log_steps == 0:
                self.print_log(mode, epoch, i, total_steps, loss.data.mean())

    def fit_styled_epoch(self, epoch: int, mode: str) -> None:
        data_loader = self.data_loader_map[mode]
        optimizer = self.optimizer_map[mode]
        total_steps = len(data_loader)
        for i, (captions, lengths) in enumerate(data_loader):
            self.model.zero_grad()
            loss = self.model(captions, lengths, mode=mode)
            loss.backward()
            optimizer.step()

            if i % self.log_steps == 0:
                self.print_log(mode, epoch, i, total_steps, loss.data.mean())

    def test(self):
        pass

    def print_log(
        self,
        mode: str,
        epoch: int,
        steps: int,
        total_steps: int,
        loss_value: float
    ) -> None:
        message = f"""
            Epoch [{epoch}/{self.num_epochs}], Mode [{mode}],
            Step [{steps}/{total_steps}], Loss: {loss_value: .4f}
        """
        print(message)

    def save(self, epoch: int) -> None:
        torch.save(self.model.state_dict(), self.save_dir / f"model-{epoch}.pth")
