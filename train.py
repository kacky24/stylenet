import os
import pickle
import torch
from torch.autograd import Variable
from build_vocab import Vocab
from data_loader import get_data_loader
from data_loader import get_styled_data_loader
from models import EncoderCNN
from models import FactoredLSTM
from loss import masked_cross_entropy


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def eval_outputs(outputs, vocab):
    # outputs: [batch, max_len - 1, vocab_size]
    indices = torch.topk(outputs, 1)[1]
    indices = indices.squeeze(2)
    indices = indices.data
    for i in range(len(indices)):
        caption = [vocab.i2w[x] for x in indices[i]]
        print(caption)

def main():
    model_path = "./pretrained_models/shuffle2/"
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # load vocablary
    with open("data/vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)

    img_path = "data/flickr7k_images"
    factual_cap_path = "data/factual_train.txt"
    humorous_cap_path = "data/humor/funny_train.txt"

    # import data_loader
    data_loader = get_data_loader(img_path, factual_cap_path, vocab, 64, shuffle=True)
    styled_data_loader = get_styled_data_loader(humorous_cap_path, vocab, 96, shuffle=True)

    # import models
    emb_dim = 300
    hidden_dim = 512
    factored_dim = 512
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
    optimizer_cap = torch.optim.Adam(cap_params, lr=0.0002)
    optimizer_lang = torch.optim.Adam(lang_params, lr=0.0005)

    # train
    total_cap_step = len(data_loader)
    total_lang_step = len(styled_data_loader)
    epoch_num = 30
    for epoch in range(epoch_num):
        # caption
        for i, (images, captions, lengths) in enumerate(data_loader):
            images = to_var(images, volatile=True)
            captions = to_var(captions.long())

            # forward, backward and optimize
            decoder.zero_grad()
            encoder.zero_grad()
            features = encoder(images)
            outputs = decoder(captions, features, mode="factual")
            loss = criterion(outputs[:, 1:, :].contiguous(), captions[:, 1:].contiguous(), lengths - 1)
            loss.backward()
            optimizer_cap.step()

            # print log
            if i % 100 == 0:
                print("Epoch [%d/%d], CAP, Step [%d/%d], Loss: %.4f"
                      % (epoch + 1, epoch_num, i, total_cap_step, loss.data[0]))

        eval_outputs(outputs, vocab)
            
        # language
        for i, (captions, lengths) in enumerate(styled_data_loader):
            captions = to_var(captions.long())

            # forward, backward and optimize
            decoder.zero_grad()
            outputs = decoder(captions, mode='humorous')
            loss = criterion(outputs, captions[:, 1:].contiguous(), lengths - 1)
            loss.backward()
            optimizer_lang.step()

            # print log
            if i % 10 == 0:
                print("Epoch [%d/%d], LANG, Step [%d/%d], Loss: %.4f"
                      % (epoch + 1, epoch_num, i, total_lang_step, loss.data[0]))

        # save models
        torch.save(decoder.state_dict(),
                   os.path.join(model_path, 'decoder-%d.pkl' % (epoch + 1,)))

        torch.save(encoder.state_dict(),
                   os.path.join(model_path, 'encoder-%d.pkl' % (epoch + 1,)))


if __name__ == '__main__':
    main()
