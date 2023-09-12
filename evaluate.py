

import train
from lib import utils, dataset
from lib.dataset import EPICDiff
from opt import get_opts
from train import get_latest_ckpt

from lib.dataset import MaskLoader
from lib import evaluation
import torch
import os


def main(hparams):

    assert hparams.eval2.dir_dst is not None

    vid = hparams.vid

    if hparams.eval2.dir_dst is None:
        dir_dst = os.path.join(f'outputs/{vid}-{model_type}')
    else:
        dir_dst = hparams.eval2.dir_dst

    dataset = EPICDiff(
        f'{hparams.vid}',
        root='data/',
        split='test',
        frames_dir='frames',
        scale=hparams.scale,
        with_radialdist=hparams.with_radialdist,
        cfg=hparams
    )

    ckpt_path = get_latest_ckpt(hparams.eval2.ckpt)

    model = train.init_model(ckpt_path, dataset)
    maskloader = MaskLoader(
        dataset=dataset,
        is_debug=False,
        empty_masks=True
    )

    if hasattr(model.hparams, 'model_type'):
        model_type = model.hparams.model_type
    else:
        model_type = 'neuraldiff'
        model.hparams.model_type = 'neuraldiff'

    if hparams.eval2.with_val:
        print('Predicting with validation ids.')
        img_ids = sorted(dataset.img_ids_test + dataset.img_ids_val)
    else:
        img_ids = sorted(dataset.img_ids_test)

    if hparams.eval2.is_debug:
        img_ids = img_ids[:1]

    results = evaluation.evaluate(
        dataset,
        model,
        maskloader,
        vis_i=1,
        save=True,
        save_dir=dir_dst,
        vid=vid,
        image_ids=img_ids,
    )

    if hparams.eval2.is_debug:
        results = {}

    for x in results['out']:
        del x['grid']
        for k in x:
            if type(x[k]) == torch.Tensor:
                x[k] = x[k].to(torch.float16)


    if hparams.eval2.save_cache:
        torch.save(results, os.path.join(dir_dst, 'cache.pt'))


if __name__ == '__main__':
    hparams = get_opts()
    hparams.actor_compat = False
    main(hparams)
