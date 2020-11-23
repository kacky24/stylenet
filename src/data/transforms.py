from typing import Tuple, Union

import numpy as np

import skimage.transform


class Rescale(object):
    """
    Rescale the image to a given size
    Args:
        output_size(int or tuple)
    """
    def __init__(self, output_size: Union[int, Tuple[int]]) -> None:
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image: np.ndarray) -> np.ndarray:
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
