import json

import numpy as np
import skimage.draw
from skimage.io import imshow, imread
import matplotlib.pyplot as plt
import torch


class VIALoaderMerged():
    def __init__(self, paths, imhw_src, imhw_dst, ignore_imhw_assert=False, each_nth=1, **kwargs):
        self.loaders = []
        for p in paths:
            self.loaders.append(VIALoader(p, imhw_src, imhw_dst, ignore_imhw_assert, each_nth=each_nth))

        self.fn_available = []
        for loader in self.loaders:
            for fn in loader.fn_available:
                self.fn_available.append(fn)
        self.fn_available = sorted(self.fn_available)

    def load_labels(self, fn):
        n_loaders = 0
        for loader in self.loaders:
            if fn in loader.fn_available:
                labels = loader.load_labels(fn)
                n_loaders += 1
        # if loaders merged, then their IDs should not intersect
        assert n_loaders == 1
        return labels


class VIALoader():
    def __init__(self, path_json, imhw_src, imhw_dst=[270, 480], ignore_imhw_assert=False, each_nth=1, **kwargs):

        with open(path_json, "r") as f:
            data = json.load(f)

        self.data = data

        # resolution used for annotating not stored in VIA, defined here as 'src'
        self.imhw_src = imhw_src
        self.imhw_src_est = [0, 0]

        # rescale polygons to 'dst'
        self.imhw_dst = imhw_dst

        if len(self.data['attribute']) == 0:
            self.labels = {}
        else:
            self.labels = {str(int(k) + 1): v for k, v in data['attribute']['1']['options'].items()}

        # via ID and filename
        self.vid2fn = {k: data['file'][k]['fname'] for k in data['file']}
        self.fn2vid = {v: k for k, v in self.vid2fn.items()}

        self.vid_available = sorted([*{*[data['metadata'][k]['vid'] for k in data['metadata']]}])
        self.fn_available = sorted([*{*[self.vid2fn[vid] for vid in self.vid_available]}])

        self.fn_available = self.fn_available[::each_nth]

        # meta data ID
        self.mid2vid = {k: data['metadata'][k]['vid'] for k in data['metadata']}
        self.vid2mid = {k: [] for k in self.vid2fn}
        for mid in self.mid2vid:
            vid = self.mid2vid[mid]
            self.vid2mid[vid].append(mid)

        # init src resolution
        for fn in self.fn_available:
            self.load_labels(fn)

        if ignore_imhw_assert:
            input('Ignoring imhw assert.')
        else:
            if self.imhw_src != [270, 480]:
                assert self.imhw_src[0] == self.imhw_src_est[0], [self.imhw_src[0], self.imhw_src_est[0]]
                assert self.imhw_src[1] == self.imhw_src_est[1], [self.imhw_src[1], self.imhw_src_est[1]]

    def load_polygon(self, mid, estimate_shape=[]):
        xy = self.data['metadata'][mid]['xy'][1:]
        xy = [(x, y) for x, y in zip(xy[1::2], xy[0::2])]
        pg = np.array(xy)
        self.imhw_src_est[0] = round(max(self.imhw_src_est[0], pg[:, 0].max()))
        self.imhw_src_est[1] = round(max(self.imhw_src_est[1], pg[:, 1].max()))
        return pg

    def draw_polygon(self, pg, im=None):
        if im is None:
            im = np.zeros(self.imhw_dst, 'bool')
        rr, cc = skimage.draw.polygon(
            pg[:,0] / self.imhw_src[0] * self.imhw_dst[0],
            pg[:,1] / self.imhw_src[1] * self.imhw_dst[1],
            im.shape
            )
        im[rr,cc] = 1
        return im

    def load_mask(self, fn):
        im_labeled = self.load_labels(fn)
        mask = im_labeled > 0
        mask = mask.astype('uint8')
        return mask

    def load_labels(self, fn):
        im_labeled = np.zeros(self.imhw_dst, dtype='uint8')
        # fn = fn.replace('jpg', 'bmp')
        try:
            vid = self.fn2vid[fn]
        except:
            vid = self.fn2vid[fn.replace('jpg', 'bmp')]
        for mid in self.vid2mid[vid]:
            if len(self.labels) == 0:
                label = 1
            else:
                if len(self.data['metadata'][mid]['av']) == 0:
                    label = 0
                else:
                    label = int(self.data['metadata'][mid]['av']['1'])
            pg = self.load_polygon(mid)
            dpg = self.draw_polygon(pg)
            im_labeled[dpg] = label + 1

        return torch.from_numpy(im_labeled)

def main():
    vl = VIALoader('annotations/P01_3objects.json')
    fn = vl.fn_available[0]
    imshow(vl.load_labels(fn))
    plt.show()
    imshow(vl.load_mask(fn))
    plt.show()
