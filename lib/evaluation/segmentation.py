"""
    Evaluate segmentation capacity of model via mAP,
    also includes renderings of segmentations and PSNR evaluation.
"""
import os
from collections import defaultdict

import git
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
import torch
import tqdm
from ..dataset.annotations import blend_mask
from PIL import Image
from sklearn.metrics import average_precision_score

from . import metrics, utils
from ..exceptions import *

def np2pt(x):
    return torch.from_numpy(x) if type(x) == np.ndarray else x


def images2grid(
    img_gt,
    img_pred,
    img_pred_static,
    img_pred_transient,
    img_pred_person,
    mask_pred,
    mask_transient,
    mask_person,
    depth=None,
):

    if depth is None:
        depth = torch.zeros_like(img_gt)
    bin2rgb = lambda x: np2pt(x)[..., None].expand(*x.shape, 3)
    row1 = torch.cat([img_gt, img_pred, depth], axis=1)
    row2 = torch.cat(
        [np2pt(x) for x in [img_pred_static, img_pred_transient, img_pred_person]],
        axis=1,
    )
    row3 = torch.cat(
        [bin2rgb(x) for x in [mask_pred, mask_transient, mask_person]], axis=1
    )
    rows = torch.cat([row1, row2, row3], axis=0).clamp(0, 1)
    im = Image.fromarray((rows * 255).byte().numpy())
    if im.size != (720, 405):
        im = im.resize((720, 405))
    return im


def results2masks(results, w=None, h=None, as_image=True):


    if as_image:
        x2im = lambda x: utils.x2im(x, w, h, type_out="pt")
    else:
        x2im = lambda x: x

    output_person = "_rgb_fine_person" in results
    output_transient = "_rgb_fine_transient" in results

    mask_stat = x2im(results["_rgb_fine_static"][:, 3])
    if output_transient:
        mask_transient = x2im(results["_rgb_fine_transient"][:, 4])
        mask_pred = mask_transient
        if output_person:
            mask_person = x2im(results["_rgb_fine_person"][:, 5])
            mask_pred = mask_pred + mask_person
        else:
            mask_person = np.zeros_like(mask_transient)

    return mask_stat, mask_transient, mask_person, mask_pred


def results2staticpsnr(results, rgbs, mask_targ, img_h=None, img_w=None, visualise=False, as_image=True):
    if as_image:
        x2y = lambda x: utils.x2im(x, w, h, type_out="pt")
    else:
        x2y = lambda x: x

    masked_pred = x2y(results['rgb_fine'][:, :3]).clone()
    masked_gt = x2y(rgbs).clone()
    masked_pred[mask_targ] = 0
    masked_gt[mask_targ] = 0
    # pytorch bug for logical not, use numpy
    # see https://github.com/pytorch/pytorch/issues/32094
    assert type(mask_targ) != torch.Tensor
    mask_static = ~(mask_targ)

    psnr = metrics.psnr(masked_pred[mask_static], masked_gt[mask_static]).item()
    if visualise and as_image:
        f, ax = plt.subplots(1, 2, figsize=(10, 10))
        plt.title(f'static PSNR: {psnr}')
        ax[0].imshow(masked_pred)
        ax[0].axis('off')
        ax[1].imshow(masked_gt)
        ax[1].axis('off')
    return psnr


def results2images(results, w, h):
    images = {}
    x2im = lambda x: utils.x2im(x, w, h, type_out="pt")

    output_person = "_rgb_fine_person" in results
    output_transient = "_rgb_fine_transient" in results

    img_pred = x2im(results["rgb_fine"])

    mask_stat, mask_transient, mask_person, mask_pred = results2masks(results, w, h)

    img_pred_static = x2im(results["rgb_fine_static"][:, :3])
    img_pred_transient = x2im(results["_rgb_fine_transient"][:, :3])
    if output_person:
        img_pred_person = x2im(results["_rgb_fine_person"][:, :3])
    else:
        img_pred_person = torch.zeros_like(img_pred_transient)

    return (
        img_pred,
        img_pred_static,
        img_pred_transient,
        img_pred_person,
        mask_pred,
        mask_transient,
        mask_person,
    )


def evaluate_sample(
    dataset,
    sample_id,
    t=None,
    visualise=False,
    gt_masked=None,
    model=None,
    mask_targ=None,
    save=False,
    pose=None,
    is_debug=False
):
    """
    Evaluate one sample of a dataset (dataset). Calculate PSNR and mAP,
    and visualise different model components for this sample. Additionally,
    1) a different timestep (`t`) can be chosen, which can be different from the
    timestep of the sample (useful for rendering the same view over different
    timesteps).
    """
    if pose is None:
        sample = dataset[sample_id]
    else:
        sample = dataset.__getitem__(sample_id, pose)
    results = model.render(sample, t=t)

    w, h = tuple(sample["img_wh"].numpy())

    (
        img_pred,
        img_pred_static,
        img_pred_transient,
        img_pred_person,
        mask_pred,
        mask_transient,
        mask_person,
    ) = results2images(results, w, h)

    img_gt = utils.x2im(sample["rgbs"], w, h, type_out="pt")

    if mask_targ is not None:
        mask_pred = resize(mask_pred, mask_targ.shape)
        average_precision = 100 * average_precision_score(
            mask_targ.reshape(-1), mask_pred.reshape(-1)
        )

    psnr = metrics.psnr(img_pred, img_gt).item()
    psnr_static = metrics.psnr(img_pred_static, img_gt).item()

    if mask_targ is not None:
        gt_masked = np2pt(blend_mask(img_gt.numpy() * 255, resize(mask_targ, img_gt.shape[:2])))
    else:
        gt_masked = img_gt

    title = f"Sample: {sample_id}. PSNR: {psnr:.2f}. "
    if mask_targ is not None:
        title += f"mAP: {average_precision:.2f}."

    depth = utils.visualize_depth(results["depth_fine"].view(h, w))  # (3, H, W)
    depth = depth.permute(1, 2, 0)

    grid = np.array(
        images2grid(
            *[np2pt(resize(x, (h, w))) for x in [gt_masked,
            img_pred,
            img_pred_static,
            img_pred_transient,
            img_pred_person,]],
            mask_pred,
            mask_transient,
            mask_person,
            depth=np2pt(resize(depth, (h, w)))
        )
    )

    grid = utils.add_title(grid, title)

    if visualise:
        if not save:
            plt.imshow(grid)
            plt.axis("off")
            plt.tight_layout()
            plt.box(False)
            plt.show()

    if not is_debug:
        results = {}

    results["grid"] = grid
    results["im_tran"] = img_pred_transient
    results["im_stat"] = img_pred_static
    results["im_pred"] = img_pred
    results["im_targ"] = img_gt
    results["psnr"] = psnr
    results["mask_pred"] = mask_pred
    results["mask_pers"] = mask_person
    results["im_pers"] = img_pred_person
    results["mask_tran"] = mask_transient
    results["sample_id"] = sample_id
    results["depth"] = np2pt(resize(depth, (h, w)))
    if mask_targ is not None:
        results["average_precision"] = average_precision

    for k in results:
        if k == "grid":
            continue
        if type(results[k]) == torch.Tensor:
            results[k] = results[k].to("cpu")

    return results


def evaluate(
    dataset,
    model,
    mask_loader,
    vis_i=5,
    save_dir="results/test",
    save=False,
    vid=None,
    epoch=None,
    timestep_const=None,
    image_ids=None,
):
    """
    Like `evaluate_sample`, but evaluates over all selected image_ids.
    Saves also visualisations and average scores of the selected samples.
    """

    results = {
        k: []
        for k in [
            "avgpre",
            "psnr",
            "masks",
            "out",
            "hp",
            "sample_ids"
        ]
    }

    if save:
        os.makedirs(f"{save_dir}/per_sample", exist_ok=False)
        p = f"{save_dir}/im/"
        os.makedirs(p, exist_ok=False)


    for i, sample_id in utils.tqdm(enumerate(image_ids), total=len(image_ids)):

        do_visualise = i % vis_i == 0


        try:
            mask_targ = mask_loader[sample_id]
            tqdm.tqdm.write(f"Test sample {i}. Frame {sample_id}.")
        except MaskNotFoundError as e:
            # print(f"Mask not found for frame {sample_id}, skipping.")
            continue

        # ignore evaluation if no mask available
        if mask_targ.sum() == 0:
            print(f"No annotations for frame {sample_id}, skipping.")
            continue

        results["hp"] = model.hparams
        results["hp"]["git_eval"] = git.Repo(
            search_parent_directories=True
        ).head.object.hexsha

        if timestep_const is not None:
            timestep = sample_id
            sample_id = timestep_const
        else:
            timestep = sample_id
        out = evaluate_sample(
            dataset,
            sample_id,
            model=model,
            t=timestep,
            visualise=do_visualise,
            mask_targ=mask_targ,
            save=save,
        )

        mask_pred = out["mask_pred"]

        if save and do_visualise:
            # results_im = utils.plt_to_im(out["grid"])
            path = f"{save_dir}/per_sample/{sample_id}.jpg"
            plt.imsave(path, out["grid"])

            plt.imsave(f"{p}/{sample_id}-mask_targ.jpg", mask_targ, cmap="gray")
            plt.imsave(f"{p}/{sample_id}-mask_pred.jpg", mask_pred, cmap="gray")
            plt.imsave(f"{p}/{sample_id}-im_targ.jpg", out["im_targ"].numpy())
            plt.imsave(
                f"{p}/{sample_id}-im_pred.jpg", out["im_pred"].clamp(0, 1).numpy()
            )

        results["avgpre"].append(out["average_precision"])

        results["psnr"].append(out["psnr"])
        results["masks"].append([mask_targ, mask_pred])
        results["out"].append(out)
        results["sample_ids"].append(out["sample_id"])

    metrics_ = {
        "avgpre": {},
        "psnr": {},
    }
    for metric in metrics_:
        metrics_[metric] = np.array(
            [x for x in results[metric] if not np.isnan(x)]
        ).mean()

    results["metrics"] = metrics_

    if save:
        with open(f"{save_dir}/metrics.txt", "a") as f:
            lines = utils.write_summary(results)
            f.writelines(f"Epoch: {epoch}.\n")
            f.writelines(lines)

    print(f"avgpre: {results['metrics']['avgpre']}, PSNR: {results['metrics']['psnr']}")

    return results
