import random
import sys
import warnings
import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import skimage.transform
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
import shutil
import tarfile
import cv2 as cv
import numpy as np
import io
import os
from . import model


SUBPLOT_FNS = [
    lambda z: z.set_xticks([], []),
    lambda z: z.set_yticks([], []),
    lambda z: z.patch.set_edgecolor('black'),
    lambda z: z.patch.set_linewidth('1'),
]


def tar2bytearr(tar_member):
    return np.asarray(
        bytearray(
            tar_member.read()
        ),
        dtype=np.uint8
    )


def set_plt_colour(mode='light'):
    import matplotlib
    if mode == 'dark':
        colour = '#ffffff'
    else:
        colour = '#000000'
    matplotlib.rcParams['axes.edgecolor'] = colour
    matplotlib.rcParams['text.color'] = colour
    matplotlib.rcParams['xtick.color'] = colour
    matplotlib.rcParams['ytick.color'] = colour


def adjust_subplots(ax, fns=SUBPLOT_FNS, exclude=None):
    """ example input:
    f, ax = subplots(2, 2)
    fns = [
        lambda z: z.set_xticks([], []),
        lambda z: z.set_yticks([], []),
        lambda z: z.patch.set_edgecolor('black'),
        lambda z: z.patch.set_linewidth('1'),
    ]
    exclude = [[0,0]]
    """
    for fn in fns:
        for index, x in np.ndenumerate(ax):
            if exclude is None:
                fn(ax[index])
            else:
                if list(index) not in exclude:
                    fn(ax[index])


def remove_mask_channels(results):
    assert all(results[k].shape[1] == 6 for k in results if 'rgb_' in k)
    results_reduced = results.copy()
    for k in results_reduced:
        if 'rgb_' in k:
            assert len(results_reduced[k].shape) == 2
            results_reduced[k] = results[k][:, :3].clone()
    assert all(results_reduced[k].shape[1] == 3 for k in results if 'rgb_' in k)
    return results_reduced


def x2im(x, w, h, type_out='np'):
    """Convert numpy or torch tensor to numpy or torch 'image'."""
    if len(x.shape) == 2 and x.shape[1] >= 3:
        x = x[:, :3]
        x = x.reshape(h, w, 3)
    else:
        x = x.reshape(h, w)
    if type(x) == torch.Tensor:
        x = x.detach().cpu()
        if type_out == "np":
            x = x.numpy()
    elif type(x) == np.array:
        if type_out == "pt":
            x = torch.from_numpy(x)
    return x


def tqdm(x, **kwargs):
    import sys

    if "ipykernel_launcher.py" in sys.argv[0]:
        # tqdm from notebook
        from tqdm.notebook import tqdm
    else:
        # otherwise default tqdm
        from tqdm import tqdm
    return tqdm(x, **kwargs)


def title2im(title, height=40, width=720, font_size=16):
    font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", font_size, encoding="unic")
    im = Image.fromarray(np.zeros([height, width, 3], dtype='uint8'))
    im_edit = ImageDraw.Draw(im)
    im_edit.text((15,15), title, (255, 255, 255), font=font)
    return np.array(im)


def add_title(im, title):
    title_as_image = title2im(title)
    im_with_title = np.concatenate([title_as_image, im], axis=0)
    return im_with_title


def set_deterministic(seed=0):

    import random

    import numpy
    import torch

    torch.manual_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)
    torch.backends.cudnn.benchmark = False


def hash_model(model, length=20):
    rstate = random.getstate()
    random.seed(1)
    s = (str([p.max().item() for p in model.parameters()]))
    s = s.replace(',', '').replace('.', '').replace(' ', '')
    assert length <= len(s)
    h = random.sample(s, length)
    # keep previous state of rng, e.g.:
    # seed(0), hash_model(model, 10), randint(0,200) == seed(0), randint(0,200)
    random.setstate(rstate)
    return ''.join(h)


def adjust_jupyter_argv():
    sys.argv = sys.argv[:1]


def write_mp4(name, frames, fps=10, format_='mp4'):
    imageio.mimwrite(name + '.' + format_, frames, format_, fps=fps)


class ImageReader:
    def __init__(self, src, scale=1, cv_flag=cv.IMREAD_UNCHANGED):
        # src can be directory or tar file

        self.scale = 1
        self.cv_flag = cv_flag

        if os.path.isdir(src):
            self.src_type = 'dir'
            self.fpaths = sorted(glob(os.path.join(src, '*')))
        elif os.path.isfile(src) and os.path.splitext(src)[1] == '.tar':
            self.tar = tarfile.open(src)
            self.src_type = 'tar'
            self.fpaths = sorted([x for x in self.tar.getnames() if 'frame_' in x and '.jpg' in x])
        else:
            print('Source has unknown format.')
            exit()

    def __getitem__(self, k):
        if self.src_type == 'dir':
            im = cv.imread(k, self.cv_flag)
        elif self.src_type == 'tar':
            member = self.tar.getmember(k)
            tarfile = self.tar.extractfile(member)
            byte_array = tar2bytearr(tarfile)
            im = cv.imdecode(byte_array, self.cv_flag)
        if self.scale != 1:
            im = cv.resize(
                im, dsize=[im.shape[0] // self.scale, im.shape[1] // self.scale]
            )
        if self.cv_flag != cv.IMREAD_GRAYSCALE:
            im = im[..., [2, 1, 0]]
        return im

    def save(self, k, dst):
        fn = os.path.split(k)[-1]
        if self.src_type == 'dir':
            shutil.copy(fn, os.path.join(dst, fn))
        elif self.src_type == 'tar':
            self.tar.extract(self.tar.getmember(k), dst)


def overlay_image(im, im_overlay, coord=(100, 70)):
    # assumes that im is 3 channel and im_overlay 4 (with alpha)
    alpha = im_overlay[:, :, 3]
    offset_rows = im_overlay.shape[0]
    offset_cols = im_overlay.shape[1]
    row = coord[0]
    col = coord[1]
    im[row : row + offset_rows, col : col + offset_cols, :] = (
        1 - alpha[:, :, None]
    ) * im[row : row + offset_rows, col : col + offset_cols, :] + alpha[
        :, :, None
    ] * im_overlay[
        :, :, :3
    ]
    return im


def get_parameters(models):
    """Get all model parameters recursively."""
    parameters = []
    if isinstance(models, list):
        for model in models:
            parameters += get_parameters(model)
    elif isinstance(models, dict):
        for model in models.values():
            parameters += get_parameters(model)
    else:
        # single pytorch model
        parameters += list(models.parameters())
    return parameters


def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    x = depth.cpu().numpy()
    x = np.nan_to_num(x)  # change nan to 0
    mi = np.min(x)  # get minimum depth
    ma = np.max(x)
    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = torchvision.transforms.ToTensor()(x_)  # (3, H, W)
    return x_


def assign_appearance(ids_train, ids_unassigned):
    # described in experiments, (3) NeRF-W: reassign each test embedding to closest train embedding
    ids = sorted(ids_train + ids_unassigned)
    g = {}
    for id in ids_unassigned:
        pos = ids.index(id)
        if pos == 0:
            # then only possible to assign to next embedding
            id_reassign = ids[1]
        elif pos == len(ids) - 1:
            # then only possible to assign to previous embedding
            id_reassign = ids[pos - 1]
        else:
            # otherwise the one that is closes according to frame index
            id_prev = ids[pos - 1]
            id_next = ids[pos + 1]
            id_reassign = min(
                (abs(ids[pos] - id_prev), id_prev), (abs(ids[pos] - id_next), id_next)
            )[1]
        g[ids[pos]] = id_reassign
    return g


### from NeRF PyTorch repo, spiral rendering, slightly adjusted
### https://github.com/yenchenlin/nerf-pytorch

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    # hwf = c2w[:,4:5]

    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(viewmatrix(z, up, c))
        # render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses


def normalize(x):
    return x / np.linalg.norm(x)


def render_poses(poses, bds, focal, path_zflat=False, shrink_factor=.8, zdelta_factor=.2):

    c2w = poses_avg(poses)
    print('recentered', c2w.shape)
    print(c2w[:3,:4])

    ## Get spiral
    # Get average pose
    up = normalize(poses[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    close_depth, inf_depth = bds.min()*.9, bds.max()*5.
    dt = .75
    mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
    focal = mean_dz

    # Get radii for spiral path
    # shrink_factor = .8
    print(shrink_factor, zdelta_factor)
    # zdelta = close_depth * .2
    zdelta = close_depth * zdelta_factor
    tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
    rads = np.percentile(np.abs(tt), 90, 0)
    c2w_path = c2w
    N_views = 120
    N_rots = 2
    if path_zflat:
#             zloc = np.percentile(tt, 10, 0)[2]
        zloc = -close_depth * .1
        # import ext
        # ext.set(c2w_path, zloc)
        # c2w_path[:3,3] = c2w_path[:3,3] + zloc * c2w_path[:3,2]
        c2w_path[:3,3] = (torch.from_numpy(c2w_path[:3,3]) + zloc * torch.from_numpy(c2w_path[:3,2])).numpy()
        rads[2] = 0.
        N_rots = 1
        # N_views /= 2
        N_views = int(N_views / 2)

    # Generate poses for spiral path
    render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)


    render_poses = np.array(render_poses).astype(np.float32)

    return render_poses


def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w


def normalize(x):
    return x / np.linalg.norm(x)