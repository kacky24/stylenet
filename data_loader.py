import os
import re
import pickle
import nltk
import skimage.io
import skimage.transform
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from build_vocab import Vocab


class Flickr7kDataset(Dataset):
    '''Flickr7k dataset'''
    def __init__(self, img_dir, caption_file, vocab, transform=None):
        '''
        Args:
            img_dir: Direcutory with all the images
            caption_file: Path to the factual caption file
            vocab: Vocab instance
            transform: Optional transform to be applied
        '''
        self.img_dir = img_dir
        self.imgname_caption_list = self._get_imgname_and_caption(caption_file)
        self.vocab = vocab
        self.transform = transform

    def _get_imgname_and_caption(self, caption_file):
        '''extract image name and caption from factual caption file'''
        with open(caption_file, 'r') as f:
            res = f.readlines()

        imgname_caption_list = []
        r = re.compile(r'#\d*')
        for line in res:
            img_and_cap = r.split(line)
            img_and_cap = [x.strip() for x in img_and_cap]
            imgname_caption_list.append(img_and_cap)

        return imgname_caption_list

    def __len__(self):
        return len(self.imgname_caption_list)

    def __getitem__(self, ix):
        '''return one data pair (image and captioin)'''
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
        caption.append(self.vocab('<s>'))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab('</s>'))
        caption = torch.Tensor(caption)
        return image, caption


class FlickrStyle7kDataset(Dataset):
    '''Styled caption dataset'''
    def __init__(self, caption_file, vocab):
        '''
        Args:
            caption_file: Path to styled caption file
            vocab: Vocab instance
        '''
        self.caption_list = self._get_caption(caption_file)
        self.vocab = vocab

    def _get_caption(self, caption_file):
        '''extract caption list from styled caption file'''
        with open(caption_file, 'r') as f:
            caption_list = f.readlines()

        caption_list = [x.strip() for x in caption_list]
        return caption_list

    def __len__(self):
        return len(self.caption_list)

    def __getitem__(self, ix):
        caption = self.caption_list[ix]
        # convert caption to word ids
        r = re.compile("\.")
        tokens = nltk.tokenize.word_tokenize(r.sub("", caption).lower())
        caption = []
        caption.append(self.vocab('<s>'))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab('</s>'))
        caption = torch.Tensor(caption)
        return caption


def get_data_loader(img_dir, caption_file, vocab, batch_size,
                    transform=None, shuffle=False, num_workers=0):
    '''Return data_loader'''
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


def get_styled_data_loader(caption_file, vocab, batch_size,
                           shuffle=False, num_workers=0):
    '''Return data_loader for styled caption'''
    flickr_styled_7k = FlickrStyle7kDataset(caption_file, vocab)

    data_loader = DataLoader(dataset=flickr_styled_7k,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers,
                             collate_fn=collate_fn_styled)
    return data_loader


class Rescale:
    '''Rescale the image to a given size
    Args:
        output_size(int or tuple)
    '''
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        image = skimage.transform.resize(image, (new_h, new_w))

        return image


def collate_fn(data):
    '''create minibatch tensors from data(list of tuple(image, caption))'''
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # images : tuple of 3D tensor -> 4D tensor
    images = torch.stack(images, 0)

    # captions : tuple of 1D Tensor -> 2D tensor
    lengths = torch.LongTensor([len(cap) for cap in captions])
    captions = [pad_sequence(cap, max(lengths)) for cap in captions]
    captions = torch.stack(captions, 0)

    return images, captions, lengths


def collate_fn_styled(captions):
    captions.sort(key=lambda x: len(x), reverse=True)

    # tuple of 1D Tensor -> 2D Tensor
    lengths = torch.LongTensor([len(cap) for cap in captions])
    captions = [pad_sequence(cap, max(lengths)) for cap in captions]
    captions = torch.stack(captions, 0)

    return captions, lengths


def pad_sequence(seq, max_len):
    seq = torch.cat((seq, torch.zeros(max_len - len(seq))))
    return seq


if __name__ == "__main__":
    with open("data/vocab.pkl", 'rb') as f:
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
