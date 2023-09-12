import train
from lib import utils, dataset
from lib.dataset import EPICDiff
from opt import get_opts
from pathlib import Path
import os
import torch

from lib.dataset import MaskLoader
from lib import evaluation

from tqdm import tqdm
from lib.evaluation.segmentation import evaluate_sample
import matplotlib.pyplot as plt

from train import get_opts
from train import get_latest_ckpt
import numpy as np
import imageio


def write_mp4(name, frames, fps=10, format_='mp4'):
    imageio.mimwrite(name + '.' + format_, frames, format_, fps=fps)


def split_list(l, num_chunks=1):
    split = []
    sz_split = len(l) // num_chunks
    for i in range(0, len(l), sz_split):
        split += [l[i:i+sz_split]]
    return split


def init(hparams):

    model_type = hparams.model_type
    vid = hparams.vid

    scale = 1

    dataset = EPICDiff(
        f'{vid}',
        root='data/',
        split='test',
        frames_dir='frames',
        scale=scale,
        with_radialdist=hparams.with_radialdist,
        cfg=hparams
    )

    # ckpt = get_latest_ckpt(f'ckpts_nle/{vid}-{model_type}')
    ckpt = get_latest_ckpt(f'ckpts/train_new_all/{vid}-{model_type}')
    model = train.init_model(ckpt, dataset, hparams=hparams)

    return model, dataset


def render(model, dataset, hparams):

    maskloader = MaskLoader(
        dataset=dataset,
        is_debug=False,
        empty_masks=True
    )

    if hparams.render.num_frames < 1:
        indices = [i for i in range(len(dataset.img_ids))]
    else:
        indices = np.linspace(
            10,
            len(dataset.img_ids) - 1,
            hparams.render.num_frames
        ).round().astype(int)

    image_ids = [dataset.img_ids[x] for x in indices]
    image_ids_all = image_ids
    if hparams.render.debug:
        image_ids = image_ids[:3]

    if hparams.render.num_chunks is not None:
        assert hparams.render.chunk_id is not None
        assert hparams.render.chunk_id <= hparams.render.num_chunks
        assert hparams.render.chunk_id > 0
        chunk_id = hparams.render.chunk_id
        image_ids = split_list(image_ids, hparams.render.num_chunks)[chunk_id - 1]
        print(image_ids, len(image_ids), len(image_ids) / len(image_ids_all))
        # input()


    if hparams.render.view_id is None:
        results = evaluation.evaluate(
            dataset,
            model,
            mask_loader=maskloader,
    #         vis_i=100,
            vid=hparams.vid,
            image_ids=image_ids,
        )
        outs = results['out']
    else:
        outs = []
        for i in tqdm(image_ids):
            out = evaluate_sample(
                dataset,
                hparams.render.view_id,
                t = i,
                model=model,
            )
            outs.append(out)

    for out in outs:
        del out['grid']

    for out in outs:
        for k in out:
            if type(out[k]) == torch.Tensor:
                out[k] = out[k].to(torch.float16)

    return outs


def save(hparams, outs):

    dir_dst = Path(hparams.render.dir_dst) / f'{hparams.vid}-{hparams.model_type}'
    os.makedirs(dir_dst, exist_ok=True)
    chunk_id = hparams.render.chunk_id
    if hparams.render.num_chunks is not None:
        str_chunks = f'-{chunk_id:0>2}_{hparams.render.num_chunks:0>2}'
    else:
        str_chunks = ''
    # print(str_chunks)
    # print(f'cache-none{str_chunks}.pt')
    # input()
    if hparams.render.view_id is None:
        torch.save(outs, dir_dst / f'cache-none{str_chunks}.pt')
    else:
        torch.save(outs, dir_dst / f'cache-{hparams.render.view_id}{str_chunks}.pt')
    # z = torch.load(dir_dst / 'cache.pt')


if __name__ == '__main__':
    hparams = get_opts()
    hparams.actor_compat = False
    model, dataset = init(hparams)
    outs = render(model, dataset, hparams)
    save(hparams, outs)