
import torch
from sklearn.metrics import average_precision_score
from .metrics import psnr
from . import segmentation
from . import metrics
import matplotlib.pyplot as plt

def test_val_sampled(ds, model, idx=4):
    n_samples = int(ds.rays.shape[0] / 5)
    rays = ds.rays[n_samples * idx: n_samples * (idx + 1)]
    ts = ds.ts[n_samples * idx: n_samples * (idx + 1), 0]
    rgbs = ds.rgbs[n_samples * idx: n_samples * (idx + 1)]
    masks = ds.masks[n_samples * idx: n_samples * (idx + 1)]
    with torch.no_grad():
        results = model(rays, ts, test_time=True)

    img_pred = results['rgb_fine'][:, :3]
    img_gt = rgbs

    psnr = metrics.psnr(img_pred, img_gt).item()

    f, ax = plt.subplots(1, 2, figsize=(10, 10))
    plt.title(f'PSNR: {psnr}')
    ax[0].imshow(img_gt.view(ds.img_h, ds.img_w, 3))
    ax[0].axis('off')
    ax[1].imshow(img_pred.view(ds.img_h, ds.img_w, 3))
    ax[1].axis('off')

    mask_pred = segmentation.results2masks(results, ds.img_w, ds.img_h)[-1]
    mask_targ = masks.view(ds.img_h, ds.img_w)

    average_precision = 100 * average_precision_score(
        mask_targ.reshape(-1), mask_pred.reshape(-1)
    )
    f, ax = plt.subplots(1, 2, figsize=(10, 10))
    plt.title(f'avgpre: {average_precision}')
    ax[0].imshow(mask_targ)
    ax[0].axis('off')
    ax[1].imshow(mask_pred)
    ax[1].axis('off')

    segmentation.results2staticpsnr(results, rgbs, mask_targ.numpy(), ds.img_h, ds.img_w, visualise=True)
