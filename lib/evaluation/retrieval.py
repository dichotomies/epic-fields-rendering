
from ..dataset.contrlearn.vis import *
from ..dataset.contrlearn.split import load_split
from sklearn.metrics import average_precision_score
from sklearn.metrics import jaccard_score
import os
from .utils import tqdm
from .utils import plt_to_im
import numpy as np
from ..utils import write_mp4
from ..model.retrieval import retrieve_label_3d
import time
from ..dataset.contrlearn.dino import load_features_wrapped


def calc_scores(dist2d, retr, mask_targ):
    mask_pred = -dist2d
    ap = 100 * average_precision_score(
        mask_targ.reshape(-1), mask_pred.reshape(-1)
    )
    mask_pred = retr
    iou = jaccard_score(
        mask_targ.reshape(-1),
        mask_pred.reshape(-1)
    ) * 100
    return {'ap': ap, 'iou': iou}


def calc_valid_ids(dataset):
    return [dataset.image_paths_inv[fn] for fn in dataset.label_loader.fn_available]


class SamplesCache:
    def __init__(self):
        pass


class LoaderPreloaded():
    def __init__(self, model, samples, ids):
        self.model = model
        self.ids = ids
        self.samples = samples

    def __getitem__(self, x):
        if type(x) is int:
            sid = x
        elif len(x) == 2:
            sid, use_cache = x

        if sid in self.ids:
            return self.samples[sid]
        else:
            raise IndexError("Invalid key: {}".format(sid))

class LoaderCached():
    def __init__(self, model, ids):
        self.cached = {}
        # self.cache_dir = f'/scratch/tmp/{hash(time.time())}'
        self.cache_dir = f'{model.hparams.scratch_dir}/qr3d/{hash(time.time())}'
        os.makedirs(self.cache_dir)
        self.model = model
        self.ids = ids

    def __getitem__(self, x):
        if type(x) is int:
            sid, use_cache = x, False
        elif len(x) == 2:
            sid, use_cache = x
        model = self.model
        ids = self.ids

        if sid in self.ids:
            if use_cache:
                file_path = os.path.join(self.cache_dir, str(sid) + '.pt')
                if sid in self.cached:
                    s = torch.load(self.cached[sid])
                else:
                    s = make_s(model, model.val_dataset, sid=sid, transient_only=False)
                    self.cached[sid] = file_path
                    torch.save(s, file_path)
            else:
                s = make_s(model, model.val_dataset, sid=sid, transient_only=False)
            # samples = {sid: s}
            return s
        else:
            raise IndexError("Invalid key: {}".format(sid))

from lib.dataset.contrlearn.vis import *
def render_samples(split, model, each_nth=1, preload=True, features_only=False):

    ids = [model.val_dataset.image_paths_inv[fn] for fn in model.val_dataset.label_loader.fn_available]

    ids = ids[::each_nth]

    if features_only:
        features = load_features_wrapped(model, ids)
    else:
        features = None

    if preload:

        samples = {}
        for sid in tqdm(ids):
            s = make_s(model, model.val_dataset, sid=sid, transient_only=False, features=features)
            samples[sid] = s

        samples = LoaderPreloaded(model, samples, ids)

    else:

        samples = LoaderCached(model=model, ids=ids)

    return samples


def render_samples_tsne(split, model, each_nth=1, each_nth_all=10000):
    samples = {}
    ids = model.val_dataset.img_ids[::each_nth_all]
    ids_labels = [
        model.val_dataset.image_paths_inv[fn] for fn in model.val_dataset.label_loader.fn_available
    ][::each_nth]
    # print(len(ids), len(ids_labels))
    # input()
    for sid in tqdm(ids + ids_labels):
        # print(sid)
        if sid in ids_labels:
            ignore_label=False
        else:
            ignore_label=True
        s = make_s(model, model.val_dataset, sid=sid, ignore_label=ignore_label, transient_only=False)
        z = {
            'emb': s['emb2d'],
            'rgb': s['rgbt'],
            'maskf': s['x']['mask_tran']
        }
        if not ignore_label:
            z['label'] = s['l']
        samples[sid] = z
    return samples


def evaluate(cfg, model, samples, split, vis=False, exist_ok=False, each_nth=10, margin=0.2, label_selected=None, accumulate_results=False):
    if label_selected is not None:
        input('label was selected.')
    if 'n00003dv' in cfg.vid:
        vid = cfg.vid.split('-')[0]
    else:
        vid = cfg.vid
    dir_dst = f'{cfg.eval.results_dir}/task2/{cfg.eval.exp}/{vid}/'
    os.makedirs(dir_dst, exist_ok=exist_ok)
    dir_images = os.path.join(dir_dst, 'images')
    os.makedirs(dir_images, exist_ok=exist_ok)

    scores_all = []
    valid_ids = calc_valid_ids(model.val_dataset)

    i = 0

    evaluation_ids = []

    results = {}

    for idq in tqdm(split['tr']):

        if (idq not in valid_ids) or (idq not in samples.ids):
            continue

        sq = samples[idq]

        for idr in split['te']:
            if (idr not in valid_ids) or (idr not in samples.ids):
                continue
            sr = samples[idr, True]
            for query_label in sq['l'].unique()[1:]:
                if label_selected is not None:
                    if query_label != label_selected:
                        continue

                eval_filename = f'{idq:07}-{idr:07}-{query_label:03}.jpg'
                if model.hparams.eval.retrieval.xray.active:
                    retr, quer, dist2d = retrieve_label_3d(
                        sq, sr, query_label=query_label, margin=margin,
                        eps=model.hparams.eval.retrieval.xray.eps
                    )
                else:
                    retr, quer, dist2d = retrieve_label(
                        sq, sr, query_label=query_label, vis=False, margin=margin
                    )
                mask_targ = sr['l'] == query_label
                if mask_targ.sum() == 0:
                    continue
                scores = calc_scores(dist2d, retr, mask_targ)
                evaluation_ids.append([idq, idr])
                scores_all.append(scores)
                if i % each_nth == 0:
                    f = vis_quer_retr(sq, sr, quer, retr, dist2d, figscale=1, dpi=150, scores=scores, query_label=query_label)
                    im = plt_to_im(f, is_notebook=cfg.is_notebook, show=vis)
                    if cfg.is_notebook:
                        plt.show()
                    plt.imsave(os.path.join(dir_images, eval_filename), im)
                    plt.clf()
                    plt.close()
                if accumulate_results:
                    maskq = sq['l'] == query_label
                    maskr = sr['l'] == query_label
                    results[idq, idr, query_label.item()] = {
                        'rgbtq': sq['rgbt'],
                        'rgbtr': sr['rgbt'],
                        'rgbq': sq['rgb'],
                        'rgbr': sr['rgb'],
                        'segq': sq['seg'],
                        'segr': sr['seg'],
                        'dist2d': dist2d,
                        'maskq': maskq,
                        'maskr': maskr,
                        'ap': scores['ap']
                    }

                i += 1

    scores = scores_all

    meanap = np.mean([x['ap'] for x in scores])
    meaniou = np.mean([x['iou'] for x in scores])
    with open(os.path.join(dir_dst, 'metrics.txt'), 'a') as f:
        f.writelines(f'ckpt:\t{model.hparams.eval.ckpt_path}\n')
        f.writelines(f'VID:\t{vid}\n')
        f.writelines(f'mAP:\t{meanap:.2f}\n')
        f.writelines(f'mIoU:\t{meaniou:.2f}\n')

    return scores, meanap, meaniou, evaluation_ids, results
    # return scores, meanap, meaniou, evaluation_ids, images


def calc_labels(images):
    return list(set([x[2] for x in list(images.keys())]))


def extract_videos(selected_label, dir_dst):
    # dir_dst, e.g. 'results/clf-vids/clf'

    images_by_label = [ims[x] for x in list(ims.keys()) if selected_label == x[2]]
    os.makedirs(dir_dst, exist_ok=True)
    write_mp4(f'{dir_dst}/{vid_}-l{selected_label}-clf', images_by_label, fps=3)