import pickle
import torch
from torch.autograd import Variable
from build_vocab import Vocab
from data_loader import get_data_loader
from data_loader import get_styled_data_loader
from models import EncoderCNN
from models import FactoredLSTM
from loss import masked_cross_entropy


def main():
    with open("data/vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)

    img_path = "data/flickr7k_images"
    cap_path = "data/factual_train.txt"
    styled_path = "data/humor/funny_train.txt"
    data_loader = get_data_loader(img_path, cap_path, vocab, 3)
    styled_data_loader = get_styled_data_loader(styled_path, vocab, 3)

    encoder = EncoderCNN(30)
    decoder = FactoredLSTM(30, 40, 40, len(vocab))

    if torch.cuda.is_available():
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    # for i, (images, captions, lengths) in enumerate(data_loader):
    for i, (captions, lengths) in enumerate(styled_data_loader):
        # images = Variable(images, volatile=True)
        captions = Variable(captions.long())

        if torch.cuda.is_available():
            # images = images.cuda()
            captions = captions.cuda()

        # features = encoder(images)

        outputs = decoder(captions, features=None, mode="humorous")
        print(lengths - 1)
        print(outputs)
        print(captions[:, 1:])

        loss = masked_cross_entropy(outputs, captions[:, 1:].contiguous(), lengths - 1)

        print(loss)

        break


if __name__ == '__main__':
    main()
