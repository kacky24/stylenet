import argparse
import os
import pickle

import torch

from src.data.loader import get_data_loader, get_styled_data_loader
from src.models.decoder import FactoredLSTM
from src.models.encoder import EncoderCNN
from src.training.loss import masked_cross_entropy
from src.utils.vocab import Vocab

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def eval_outputs(outputs: torch.Tensor, vocab: Vocab) -> None:
    # outputs: [batch, max_len - 1, vocab_size]
    indices = torch.topk(outputs, 1)[1]
    indices = indices.squeeze(2)
    indices = indices.data
    for i in range(len(indices)):
        caption = [vocab.i2w[x] for x in indices[i]]
        print(caption)


def main(args: argparse.Namespace) -> None:
    model_path = args.model_path
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # load vocablary
    with open(args.vocab_path, "rb") as f:
        vocab = pickle.load(f)

    img_path = args.img_path
    factual_cap_path = args.factual_caption_path
    humorous_cap_path = args.humorous_caption_path

    # import data_loader
    data_loader = get_data_loader(img_path, factual_cap_path, vocab,
                                  args.caption_batch_size, shuffle=True)
    styled_data_loader = get_styled_data_loader(humorous_cap_path, vocab,
                                                args.language_batch_size,
                                                shuffle=True)

    # import models
    emb_dim = args.emb_dim
    hidden_dim = args.hidden_dim
    factored_dim = args.factored_dim
    vocab_size = len(vocab)
    encoder = EncoderCNN(emb_dim)
    decoder = FactoredLSTM(emb_dim, hidden_dim, factored_dim, vocab_size)

    if torch.cuda.is_available():
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    # loss and optimizer
    criterion = masked_cross_entropy
    cap_params = list(decoder.parameters()) + list(encoder.A.parameters())
    lang_params = list(decoder.parameters())
    optimizer_cap = torch.optim.Adam(cap_params, lr=args.lr_caption)
    optimizer_lang = torch.optim.Adam(lang_params, lr=args.lr_language)

    # train
    total_cap_step = len(data_loader)
    total_lang_step = len(styled_data_loader)
    epoch_num = args.epoch_num
    for epoch in range(epoch_num):
        # caption
        for i, (images, captions, lengths) in enumerate(data_loader):
            # forward, backward and optimize
            decoder.zero_grad()
            encoder.zero_grad()
            features = encoder(images)
            outputs = decoder(captions, features, mode="factual")
            loss = criterion(outputs[:, 1:, :].contiguous(),
                             captions[:, 1:].contiguous(), lengths - 1)
            loss.backward()
            optimizer_cap.step()

            # print log
            if i % args.log_step_caption == 0:
                print("Epoch [%d/%d], CAP, Step [%d/%d], Loss: %.4f"
                      % (epoch + 1, epoch_num, i, total_cap_step,
                          loss.data.mean()))

        eval_outputs(outputs, vocab)

        # language
        for i, (captions, lengths) in enumerate(styled_data_loader):
            # forward, backward and optimize
            decoder.zero_grad()
            outputs = decoder(captions, mode="humorous")
            loss = criterion(outputs, captions[:, 1:].contiguous(), lengths - 1)
            loss.backward()
            optimizer_lang.step()

            # print log
            if i % args.log_step_language == 0:
                print("Epoch [%d/%d], LANG, Step [%d/%d], Loss: %.4f"
                      % (epoch + 1, epoch_num, i, total_lang_step,
                          loss.data.mean()))

        # save models
        torch.save(decoder.state_dict(),
                   os.path.join(model_path, "decoder-%d.pkl" % (epoch + 1,)))

        torch.save(encoder.state_dict(),
                   os.path.join(model_path, "encoder-%d.pkl" % (epoch + 1,)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="StyleNet: Generating Attractive Visual Captions \
                     with Styles")
    parser.add_argument("--model_path", type=str, default="pretrained_models",
                        help="path for saving trained models")
    parser.add_argument("--vocab_path", type=str, default="./data/vocab.pkl",
                        help="path for vocabrary")
    parser.add_argument("--img_path", type=str,
                        default="./data/flickr7k_images",
                        help="path for train images directory")
    parser.add_argument("--factual_caption_path", type=str,
                        default="./data/factual_train.txt",
                        help="path for factual caption file")
    parser.add_argument("--humorous_caption_path", type=str,
                        default="./data/humor/funny_train.txt",
                        help="path for humorous caption file")
    parser.add_argument("--romantic_caption_path", type=str,
                        default="./data/romantic/romanntic_train.txt",
                        help="path for romantic caption file")
    parser.add_argument("--caption_batch_size", type=int, default=64,
                        help="mini batch size for caption model training")
    parser.add_argument("--language_batch_size", type=int, default=96,
                        help="mini batch size for language model training")
    parser.add_argument("--emb_dim", type=int, default=300,
                        help="embedding size of word, image")
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="hidden state size of factored LSTM")
    parser.add_argument("--factored_dim", type=int, default=512,
                        help="size of factored matrix")
    parser.add_argument("--lr_caption", type=int, default=0.0002,
                        help="learning rate for caption model training")
    parser.add_argument("--lr_language", type=int, default=0.0005,
                        help="learning rate for language model training")
    parser.add_argument("--epoch_num", type=int, default=30)
    parser.add_argument("--log_step_caption", type=int, default=50,
                        help="steps for print log while train caption model")
    parser.add_argument("--log_step_language", type=int, default=10,
                        help="steps for print log while train language model")
    args = parser.parse_args()

    main(args)
