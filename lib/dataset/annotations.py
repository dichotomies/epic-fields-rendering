import os

import matplotlib.pyplot as plt
import torch
import numpy as np
import PIL.Image
from ..exceptions import *


def blend_mask(im, mask, colour=[1, 0, 0], alpha=0.5, show_im=False):
    """Blend an image with a mask (colourised via `colour` and `alpha`)."""
    if type(im) == torch.Tensor:
        im = im.numpy()
    im = im.copy()
    if im.max() > 1:
        im = im.astype(np.float) / 255
    for ch, rgb_v in zip([0, 1, 2], colour):
        im[:, :, ch][mask == 1] = im[:, :, ch][mask == 1] * (1 - alpha) + rgb_v * alpha
    if show_im:
        plt.imshow(im)
        plt.axis("off")
        plt.show()
    return im


class MaskLoader:
    """Loads masks for a dataset initialised with a video ID."""

    def __init__(self, dataset, is_debug=False, empty_masks=False):
        self.frames_dir = os.path.join(dataset.root, "frames")
        self.annotations_dir = os.path.join(dataset.root, "annotations")
        self.image_paths = dataset.image_paths
        self.imread = dataset.imread
        self.is_debug = is_debug
        self.empty_masks = empty_masks
        self.dataset = dataset

        if not empty_masks:
            # e.g. for case of training on static or non-annotated data
            print(f"ID of loaded scene: {dataset.vid}.")
            print(f"Number of annotations: {len(os.listdir(self.annotations_dir))}.")

    def calc_image_ids(self):
        if self.empty_masks:
            return sorted(self.image_paths.keys())
        image_ids = []
        image_paths_inv = {v.split('.')[0]: k for k, v in self.image_paths.items()}
        for fn in os.listdir(self.annotations_dir):
            x = fn.split('.')[0]
            if x in image_paths_inv:
                image_ids.append(image_paths_inv[x])
        return sorted(image_ids)

    def __getitem__(self, sample_id):
        image_id, image_ext = self.image_paths[sample_id].split(".")
        image_ext = 'bmp'

        im = self.imread(sample_id)

        try:
            if self.empty_masks:
                mask = PIL.Image.fromarray(
                    (
                        (
                            np.random.randint(0, 255, im.shape[:2]) > 200
                        ).astype('uint8')
                    )
                )
            else:
                mask = PIL.Image.open(
                        os.path.join(self.annotations_dir, image_id + "." + image_ext)
                    )
        except FileNotFoundError as e:
            raise MaskNotFoundError

        mask = np.array(mask.resize(
            (int(self.dataset.meta['image_w']),
            int(self.dataset.meta['image_h']))
        ))

        if self.is_debug:
            blend_mask(im, mask, show_im=True)
            return mask, im

        return mask
