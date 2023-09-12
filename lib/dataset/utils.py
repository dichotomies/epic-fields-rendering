import torch
from kornia import create_meshgrid

from ..utils import x2im


def fn2id(fn):
    assert 'IMG' in fn
    return int(fn.split('.')[0].split('_')[1])


def sample_validation_frame(dataset):
    ext = dataset.get_image_ext()
    idx = dataset.img_ids[len(dataset.img_ids) // 2]
    return idx


def undistort(x_dist, y_dist, radialcoeffs):
    # stabilised radial distortion
    r = (x_dist ** 2 + y_dist ** 2)

    k1 = radialcoeffs[0]
    rdist = k1 * r

    if len(radialcoeffs) == 2:
        k2 = radialcoeffs[1]
        rdist = rdist + k2 * (r ** 2)

    delta_x = x_dist * rdist
    delta_y = y_dist * rdist
    x_udst = x_dist - delta_x
    y_udst = y_dist - delta_y
    return x_udst, y_udst


def get_ray_directions(H, W, K, radialcoeffs=None):
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    u, v = grid.unbind(-1)
    fx, fy, u0, v0 = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (u - u0) / fx
    y = -(v - v0) / fy
    z = -torch.ones_like(x)

    if radialcoeffs is not None:
        x, y = undistort(x, y, radialcoeffs)

    directions = torch.stack([x, y, z], -1)

    return directions


def get_rays(directions, c2w):
    rays_d = directions @ c2w[:, :3].T
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    rays_o = c2w[:, 3].expand(rays_d.shape)
    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d
