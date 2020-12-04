import json
import logging
import sys
from pathlib import Path

import click

import torch

sys.path.append("./")
logging.basicConfig(level=logging.INFO)

from src.data.loader import get_factual_data_loader, get_styled_data_loader
from src.models import StyleNet
from src.training.trainer import Trainer
from src.utils.vocab import Vocabulary
from src.utils.vocab import build_vocab as _build_vocab


@click.group()
@click.option("--config_path", type=str, default="./config.json")
@click.pass_context
def cmd(ctx: click.Context, config_path: str):
    with open(config_path, "r") as f:
        config = json.load(f)
    ctx.obj = config


@cmd.command()
@click.pass_context
def build_vocab(ctx):
    vocab_path = ctx.obj["vocab_path"]
    caption_paths = ctx.obj["caption_paths"].copy()
    factual_caption_path = Path(caption_paths.pop("factual"))
    styled_caption_path_list = list(map(Path, caption_paths.values()))
    vocab = _build_vocab(factual_caption_path, styled_caption_path_list)
    vocab.save(vocab_path)


@cmd.command()
@click.pass_context
def train(ctx):
    config = ctx.obj
    img_dir = config["img_dir"]
    vocab_path = config["vocab_path"]
    factual_batch_size = config["factual_batch_size"]
    styled_batch_size = config["styled_batch_size"]
    emb_dim = config['emb_dim']
    hidden_dim = config["hidden_dim"]
    factored_dim = config["factored_dim"]
    num_epochs = config["num_epochs"]
    log_steps = config["log_steps"]
    model_dir = config["model_dir"]
    lr_factual = config["lr_factual"]
    lr_styled = config["lr_styled"]
    vocab = Vocabulary()
    vocab.load(vocab_path)
    caption_paths = config["caption_paths"]
    mode_list = list(caption_paths.keys())
    styled_mode_list = mode_list.copy()
    styled_mode_list.remove("factual")
    data_loader_map = {
        "factual": get_factual_data_loader(
            img_dir,
            caption_paths["factual"],
            vocab,
            factual_batch_size,
            shuffle=False
        )
    }
    for k in styled_mode_list:
        data_loader_map[k] = get_styled_data_loader(
            caption_paths[k], vocab, styled_batch_size, shuffle=True
        )
    model = StyleNet(emb_dim, hidden_dim, factored_dim, len(vocab), mode_list)
    factual_params = list(model.decoder.parameters(), model.encoder.A.parameters())
    optimizer_map = {"factual": torch.optim.Adam(factual_params, lr=lr_factual)}
    for k in styled_mode_list:
        styled_params = list(model.decoder.S_d[k].parameters())
        optimizer = torch.optim.Adam(styled_params, lr=lr_styled)
        optimizer_map[k] = optimizer

    trainer = Trainer(
        model,
        optimizer_map,
        data_loader_map,
        mode_list,
        num_epochs,
        log_steps,
        model_dir
    )
    trainer.fit()


if __name__ == "__main__":
    cmd()
