import json
import math
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd

from collections import defaultdict
from .utils import *
from ..utils import write_mp4, ImageReader
from .annotations import MaskLoader
from ..colmap_converter.meta import load_colmap, calc_meta


def tqdm(x, **kwargs):
    import sys

    if "ipykernel_launcher.py" in sys.argv[0]:
        # tqdm from notebook
        from tqdm.notebook import tqdm
    else:
        # otherwise default tqdm
        from tqdm import tqdm
    return tqdm(x, **kwargs)


def adjust_intrinsics(self):
    K = self.K.copy()
    scale = self.img_w / 228 # from initial resolution, used for COLMAP
    K[:2, :2] = K[:2, :2] * scale
    K[0,2] = self.img_w / 2
    K[1,2] = self.img_h / 2
    return K


def load_ids(dataset, split, root):
    split_all = pd.read_json(f'{root}/{split}.json', orient='index')
    split = split_all.loc[dataset.vid]
    frames = [k_ for k in split for k_ in k]
    ids = [dataset.image_paths_inv[k] for k in frames]
    return ids


def load_meta(root, name='meta.json'):
    path = os.path.join(root, name)
    with open(path, 'r') as fp:
        meta = json.load(fp)
    for k in ['nears', 'fars', 'images']:
        meta[k] = {int(i): meta[k][i] for i in meta[k]}
    meta['poses'] = {int(i): np.array(meta['poses'][i]) for i in meta['poses']}
    meta['intrinsics'] = np.array(meta['intrinsics'])
    return meta


class EmptyDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __getitem__(self, idx):
        return 0

    def __len__(self):
        return 1


class EPICDiff(Dataset):
    def __init__(self, vid, root="data", frames_dir='frames', split=None, scale=1, with_radialdist=False, init_maskloader=False, cfg=None, model=None):

        self.cfg = cfg
        self.root = os.path.join(root, vid)
        self.vid = vid
        self.ids_val = None
        self.init_maskloader = init_maskloader
        self.split = split
        self.val_num = 1
        self.transform = torchvision.transforms.ToTensor()
        self.frames_dir = frames_dir
        self.scale = scale
        self.with_radialdist = with_radialdist
        self.model = model
        self.dir_colmap_model = None
        self.init_meta()

        pid = self.vid.split('_')[0]
        src = f'data/ek100/{pid}/rgb_frames/{self.vid}.tar'
        self.imreader = ImageReader(src)

        if self.split in ['train', 'val']:
            self.init_cache()
        self.task = getattr(self.cfg.data.set, self.split[:2])['task']

    def imshow(self, index):
        plt.imshow(self.imread(index))
        plt.axis("off")
        plt.show()

    def imread(self, index, as_numpy=True):

        im = self.imreader['./' + self.image_paths[index]]
        if as_numpy:
            return np.array(im).copy()
        else:
            return im

    def x2im(self, x, type_out='np'):
        return x2im(x, self.img_w, self.img_h, type_out=type_out)

    def trid2sid(self, idx):
        return self.img_ids_train[math.floor(((idx ) / (self.img_h * self.img_w)))]

    def get_image_ext(self):
        return list(self.meta['images'].values())[0].split('.')[1]

    def sample_rays(self, idx):

        idx_sample = self.sids[idx].item()

        sample = {}
        c2w = torch.FloatTensor(self.poses_dict[idx_sample])
        img_pxl_idx = self.img_pxl_idxs[idx]
        rays_o, rays_d = get_rays(self.directions[img_pxl_idx], c2w)

        c2c = torch.zeros(3, 4).to(c2w.device)
        c2c[:3, :3] = torch.eye(3, 3).to(c2w.device)
        rays_o_c, rays_d_c = get_rays(self.directions[img_pxl_idx], c2c)

        rays_t = idx_sample * torch.ones(len(rays_o), 1).long()

        rays = torch.cat(
            [
                rays_o,
                rays_d,
                self.nears[idx_sample] * torch.ones_like(rays_o[:, :1]),
                self.fars[idx_sample] * torch.ones_like(rays_o[:, :1]),
                rays_o_c,
                rays_d_c,
            ],
            1,
        )

        sample["rays"] = rays.squeeze()
        sample["ts"] = rays_t.squeeze()

        return sample

    def rgbs_per_image(self, idx):
        img = self.imread(idx, as_numpy=False)
        img = self.transform(img)  # (3, h, w)
        rgbs = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGB
        return rgbs


    def rays_per_image(self, idx, pose=None):
        """Return sample with rays, frame index etc."""
        sample = {}
        if pose is None:
            sample["c2w"] = c2w = torch.FloatTensor(self.poses_dict[idx])
        else:
            sample["c2w"] = c2w = pose

        sample["im_path"] = self.image_paths[idx]

        directions = get_ray_directions(
            self.img_h, self.img_w, self.K, self.radialcoeffs)
        rays_o, rays_d = get_rays(directions, c2w)

        c2c = torch.zeros(3, 4).to(c2w.device)
        c2c[:3, :3] = torch.eye(3, 3).to(c2w.device)
        rays_o_c, rays_d_c = get_rays(directions, c2c)

        rays_t = idx * torch.ones(len(rays_o), 1).long()

        rays = torch.cat(
            [
                rays_o,
                rays_d,
                self.nears[idx] * torch.ones_like(rays_o[:, :1]),
                self.fars[idx] * torch.ones_like(rays_o[:, :1]),
                rays_o_c,
                rays_d_c,
            ],
            1,
        )

        sample["rays"] = rays
        sample["img_wh"] = torch.LongTensor([self.img_w, self.img_h])
        sample["ts"] = rays_t

        return sample

    def init_meta(self):
        """Load meta information, e.g. intrinsics, train, test, val split etc."""

        self.dir_colmap_model = f'data/reconstructions/{self.cfg.vid}'

        dir_frames = os.path.join(self.dir_colmap_model.split('sparse')[0], 'images')
        self.dir_frames = dir_frames

        colmap = load_colmap(self.dir_colmap_model)
        meta = calc_meta(colmap, split_nth=0)

        self.meta = meta
        self.meta['ids_all'] = sorted(self.meta['ids_all'])
        self.img_ids = meta["ids_all"]
        print('Number of images:', len(self.meta['ids_all']))
        self.img_ids_train = meta["ids_all"]

        self.image_paths = meta['images']
        self.image_paths_inv = {v: k for k, v in self.image_paths.items()}

        if self.cfg.train.ignore_split:
            print(' --- Using all image IDs for training (task 2).')
            self.img_ids_train = self.img_ids
            self.img_ids_test = self.img_ids
            self.img_ids_val = self.img_ids
        else:
            print(f'Training with {len(frames_intersection)} ({len(frames_intersection) / len(self.image_paths_inv) * 100:.2f}%) out of {len(self.image_paths_inv)} frames')
            # if VID included in split
            split_train_all = pd.read_json(f'{self.cfg.split_root}/train.json', orient='index')
            split_train = split_train_all.loc[self.vid]
            split_train_valid_mask = ~split_train.isnull()
            split_train_valid = split_train[split_train_valid_mask].tolist()
            frames_intersection = set(split_train_valid).intersection(self.image_paths_inv)
            split_ids_train = [self.image_paths_inv[x] for x in split_train_valid]
            self.img_ids_test = sorted(load_ids(self, split='test', root=self.cfg.split_root))
            self.img_ids_val = sorted(load_ids(self, split='val', root=self.cfg.split_root))
            self.img_ids_train = split_ids_train

        self.poses_dict = meta["poses"]
        self.nears = meta["nears"]
        self.fars = meta["fars"]
        if self.init_maskloader:
            self.maskloader = MaskLoader(self)
            self.ids_val = self.maskloader.calc_image_ids()[::self.cfg.eval.sampled.every_nth]
            # use img_ids_train attribute for ids_val for emulating ray-wise evaluation
            if self.ids_val is not None:
                self.img_ids_train = self.ids_val

        # downscale image and intrinsics
        self.K = meta["intrinsics"] / self.scale
        self.K[2,2] = 1
        self.img_h = int(meta['image_h'] / self.scale)
        self.img_w = int(meta['image_w'] / self.scale)

        # if self.split == 'train_new':
        #     assert self.img_w == self.meta['image_w']
        #     assert self.img_h == self.meta['image_h']

        if self.with_radialdist:
            camera = self.meta['camera']
            if camera['model'] == 'SIMPLE_RADIAL':
                f, cx, cy, k = camera['params']
                self.radialcoeffs = [k]
            elif camera['model'] == 'OPENCV':
                fx, fy, cx, cy, k1, k2, p1, p2 = camera['params']
                self.radialcoeffs = [k1, k2]
            else:
                input('Camera model unknown')
        else:
            self.radialcoeffs = None

        self.directions = get_ray_directions(
            self.img_h, self.img_w, self.K, self.radialcoeffs
        ).reshape(-1, 3)

    def init_cache(self):

        if self.split in ['train']:
            # create buffer of all rays and rgb data
            self.rgbs = []
            self.sids = []
            self.img_pxl_idxs = []

            if self.split == 'tr' and self.cfg.task == 'cli':
                self.masks_tr = []

            ids = self.img_ids_train

            indices_flat = list(range(self.img_h * self.img_w))

            for idx in tqdm(ids):

                if len(ids) > 6000:
                    sampling_idx = random.sample(indices_flat, int(20000 * 1024 / len(ids)) * 15)
                else:
                    sampling_idx = indices_flat

                self.img_pxl_idxs += [torch.arange(0, self.img_h * self.img_w)[sampling_idx]]
                self.rgbs += [self.rgbs_per_image(idx)[sampling_idx]]
                self.sids += [torch.LongTensor([idx] * self.img_h * self.img_w)[sampling_idx]]

            self.rgbs = torch.cat(self.rgbs, 0)  # ((N_images-1)*h*w, 3)
            self.sids = torch.cat(self.sids, 0)  # ((N_images-1)*h*w, 3)
            self.img_pxl_idxs = torch.cat(self.img_pxl_idxs, 0)

            self.n_train = len(self.rgbs)


    def __len__(self):
        if self.split in ['train']:
            return self.n_train
        elif self.split == "val":
            # evaluate only one image, sampled from val img ids
            return 1
        else:
            # choose any image index
            return max(self.img_ids)

    def get_train_sample(self, idx):
        sample = self.sample_rays(idx)
        if self.split in ['train']:
            rgb = self.rgbs[idx]
        sample['rgbs'] = rgb
        sample['trid'] = torch.LongTensor([idx])
        sample['sid'] = self.sids[idx]

        return sample

    def __getitem__(self, idx, pose=None):

        if self.split in ['train', 'train_new']:
            idx_rand = random.randint(0, self.n_train - 1)
            sample = self.get_train_sample(idx_rand)

        elif self.split == "val":
            # for tuning hyperparameters, tensorboard samples
            idx = sample_validation_frame(self)
            sample = self.rays_per_image(idx, pose)
            sample["rgbs"] = self.rgbs_per_image(idx)

        elif self.split == "test":
            sample = self.rays_per_image(idx, pose)
            sample["rgbs"] = self.rgbs_per_image(idx)

        else:
            # for arbitrary samples, e.g. summary video when rendering over all images
            sample = self.rays_per_image(idx, pose)
            sample["rgbs"] = self.rgbs_per_image(idx)

        return sample
