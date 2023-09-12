
import argparse
from glob import glob
import os
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import torch
import imageio


def write_mp4(name, frames, fps=10, format_='mp4'):
    imageio.mimwrite(name + '.' + format_, frames, format_, fps=fps)


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--view",
        # 15526 or none
        type=str,
    )

    parser.add_argument(
        "--vid",
        # 15526 or none
        type=str,
    )

    parser.add_argument(
        "--dir_videos",
        # ../outputs/supplementary/video_nips/
        type=str,
    )

    parser.add_argument(
        '-f',
        default='test',
        #
        type=str,
    )

    args = parser.parse_args()

    return args

def main(args):

    vid = args.vid
    view = args.view
    dir_videos = args.dir_videos

    dir_src = f'{dir_videos}/{vid}-neuraldiff/'

    dir_dst = os.path.join(dir_src, f'video-{view}')
    os.makedirs(dir_dst, exist_ok=True)

    paths = sorted([x for x in glob(dir_src + '/*') if '.pt' in x])

    paths = sorted([x for x in paths if x.split('/')[-1].split('-')[1] == view])

    cache = [torch.load(p) for p in paths[:]]

    key_mapping = {
        'im_tran': 'rgb_semistatic',
        'im_pers': 'rgb_im_dynamic',
        'im_stat': 'rgb_static',
        'im_pred': 'rgb_pred',
        'im_targ': 'rgb_target',
        'mask_tran': 'mask_semistatic',
        'mask_pers': 'mask_dynamic',
        'depth': 'depth',
    }

    for k in tqdm(key_mapping):
    # for k in tqdm(['im_pred']):
        ims = []
        for c in cache:
            for i in range(len(c)):
                # print(c[i]['sample_id'])
                ims.append(c[i][k])
                # plt.imshow(c[i][k].float())
                # plt.show()
        write_mp4(os.path.join(dir_dst, key_mapping[k]), ims)

if __name__ == '__main__':

    args = parse_args()
    main(args)
