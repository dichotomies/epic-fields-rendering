import os
from collections import defaultdict

import matplotlib.pyplot as plt
import pytorch_lightning
import torch
from torch.optim import Adam, RMSprop
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR
from torch.utils.data import DataLoader
import lib
import lib.dataset as dataset
import lib.model as model
import lib.utils as utils
import lib.evaluation as evaluation
from lib.evaluation.metrics import *
from lib.loss import Loss
from lib.loss import calc_transient_reg
from opt import get_opts
from lib.utils import *
from lib.evaluation.segmentation import evaluate_sample
from lib.dataset.utils import sample_validation_frame
import omegaconf
import warnings
import math
import warnings

import opt


def get_latest_ckpt(ckpt_path, with_print=True):
    if os.path.isdir(ckpt_path):
        # last checkpoint in folder
        if with_print:
            print('Checkpoint directory selected.')

        ckpts_mapped = {}

        ckpts = os.listdir(ckpt_path)

        for ckpt in ckpts:

            if not('-v' in ckpt.split('=')[1].split('.')[0]):
                remapped = f"{ckpt.split('=')[1].split('.')[0]}-v0"
                ckpts_mapped[f"epoch={remapped}.ckpt"] =  ckpt
            else:
                assert len(ckpt.split('=')[1].split('.')[0].split('-')[1]) == 2
                ckpts_mapped[ckpt] = ckpt

        last_ckpt = ckpts_mapped[sorted(ckpts_mapped)[-1]]

        ckpt_path = os.path.join(ckpt_path, last_ckpt)
        if with_print:
            print(f'Choosing latest checkpoint: {ckpt_path}')
    return ckpt_path


def on_load_checkpoint(self, checkpoint: dict) -> None:
    state_dict = checkpoint["state_dict"]
    model_state_dict = self.state_dict()
    is_changed = False
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print(f"Skip loading parameter: {k}, "
                            f"required shape: {model_state_dict[k].shape}, "
                            f"loaded shape: {state_dict[k].shape}")
                state_dict[k] = model_state_dict[k]
                is_changed = True
        else:
            print(f"Dropping parameter {k}")
            is_changed = True

    if is_changed:
        print('removing optimiser states and LREmbedding')
        checkpoint.pop("optimizer_states", None)


def init_model(ckpt_path, dataset=None, hparams=None, vid=None, emb_range_fix=None, max_frame_id=None):

    ckpt_path = get_latest_ckpt(ckpt_path)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    hparams_ckpt = omegaconf.DictConfig(ckpt["hyper_parameters"])

    if hparams is None:
        # if loading old model that misses new args, then add new args
        hparams = opt.get_opts(vid=vid)
        hparams = omegaconf.OmegaConf.merge(hparams, hparams_ckpt)
    else:
        # otherwise, overwrite ckpt hparams with given hparams
        hparams = omegaconf.OmegaConf.merge(hparams_ckpt, hparams)

    conf_symmetric_difference = lib.omegaconf.conf_symmetric_difference(hparams, hparams_ckpt)
    if len(conf_symmetric_difference) > 0:
        warnings.warn('Unmatched keys:' + str(conf_symmetric_difference))

    model = NeuralDiffSystem(
        hparams, train_dataset=dataset, val_dataset=dataset
    )
    try:
        model.load_state_dict(ckpt["state_dict"])
    except Exception as e:
        warnings.warn('Some model components were not loaded from checkpoint. \
            Loading with `strict=False`.'.replace('  ', '')
        )
        if hparams.debug.load_emb_different_size:
            on_load_checkpoint(model, ckpt)
        elif hparams.model_type in ['t-nerf', 'nerf-w'] and hparams.ckpt_path is not None:
            # ignores optimiser
            on_load_checkpoint(model, ckpt)
        else:
            model.load_state_dict(ckpt["state_dict"], strict=False)

    if dataset is not None:

        g_test = assign_appearance(dataset.img_ids_train, dataset.img_ids_test)
        g_val = assign_appearance(dataset.img_ids_train, dataset.img_ids_val)

        if model.hparams.model_type == 'nerf-w':
            print(' --- reassigning nerf-w latent codes')

        for g in [g_test, g_val]:
            for i, i_train in g.items():
                model.embedding_a.weight.data[i] = model.embedding_a.weight.data[
                    i_train
                ]
                if model.hparams.model_type == 'nerf-w':
                    model.embedding_t.weight.data[i] = model.embedding_t.weight.data[
                    i_train
                ]

    if model.hparams.model_type == 't-nerf':
        if max_frame_id is not None:
            model.embedding_t.set_max_frame_id(max_frame_id)
        else:
            model.embedding_t.set_max_frame_id(max(model.val_dataset.img_ids))

    return model


class NeuralDiffSystem(pytorch_lightning.LightningModule):
    def __init__(self, hparams, train_dataset=None, val_dataset=None):
        super().__init__()
        self.hparams.update(hparams)
        hparams = self.hparams

        if self.hparams.deterministic:
            utils.set_deterministic()

        # for avoiding reinitialization of dataloaders when debugging/using notebook
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.counter = 0

        self.loss = Loss(model_type=self.hparams.model_type)
        if self.hparams.contrlearn.clf.loss_type == 'l1':
            self.feature_loss = torch.nn.L1Loss()
        if self.hparams.contrlearn.clf.loss_type == 'huber':
            self.feature_loss = torch.nn.HuberLoss()

        self.triplet_loss = torch.nn.TripletMarginLoss(
            margin=hparams.contrlearn.triplet.margin,
            p=2
        )

        # pt lightning
        self.register_buffer('z_steps', torch.linspace(0, 1, hparams.N_samples))

        self.models_to_train = []
        self.embedding_xyz = model.PosEmbedding(
            hparams.N_emb_xyz - 1, hparams.N_emb_xyz
        )
        self.embedding_dir = model.PosEmbedding(
            hparams.N_emb_dir - 1, hparams.N_emb_dir
        )

        self.embeddings = {
            "xyz": self.embedding_xyz,
            "dir": self.embedding_dir,
        }

        if self.hparams.model_type == 'neuraldiff':
            self.embedding_t = model.LREEmbedding(
                N=hparams.N_vocab, D=hparams.N_tau, K=hparams.lowpass_K
            )
        elif self.hparams.model_type == 't-nerf':
            self.embedding_t = model.TimeEmbedding(
                8 - 1, 8
            )
        elif self.hparams.model_type == 'nerf-w':
            self.embedding_t = torch.nn.Embedding(hparams.N_vocab, hparams.N_tau)

        self.embeddings["t"] = self.embedding_t
        self.models_to_train += [self.embedding_t]

        self.embedding_a = torch.nn.Embedding(hparams.N_vocab, hparams.N_a)
        self.embeddings["a"] = self.embedding_a
        self.models_to_train += [self.embedding_a]

        self.nerf_coarse = model.NeuralDiff(
            "coarse",
            in_channels_xyz=6 * hparams.N_emb_xyz + 3,
            in_channels_dir=6 * hparams.N_emb_dir + 3,
            W=hparams.model_width,
            actor_compat=hparams.actor_compat,
            sz_emb=hparams.contrlearn.emb.size,
            emb_range_fix=hparams.contrlearn.clf.emb_range_fix,
        )
        self.models = {"coarse": self.nerf_coarse}
        if hparams.N_importance > 0:
            self.nerf_fine = model.NeuralDiff(
                "fine",
                in_channels_xyz=6 * hparams.N_emb_xyz + 3,
                in_channels_dir=6 * hparams.N_emb_dir + 3,
                encode_dynamic=True,
                in_channels_a=hparams.N_a,
                in_channels_t=hparams.N_tau,
                beta_min=hparams.beta_min,
                W=hparams.model_width,
                actor_compat=hparams.actor_compat,
                sz_emb=hparams.contrlearn.emb.size,
                emb_range_fix=hparams.contrlearn.clf.emb_range_fix,
            )
            self.models["fine"] = self.nerf_fine
        self.models_to_train += [self.models]

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def forward(self, rays, ts, test_time=False, with_tqdm=False):
        perturb = 0 if test_time else self.hparams.perturb
        noise_std = 0 if test_time else self.hparams.noise_std
        B = rays.shape[0]
        results = defaultdict(list)

        device_prev = self.device

        if with_tqdm:
            itx = lambda x: tqdm(x)
        else:
            itx = lambda x: x

        for i in itx(range(0, B, self.hparams.chunk)):
            if test_time:
                device='cuda:0'
                # device='cpu'
                rays_ = rays[i : i + self.hparams.chunk].detach().clone().to(device)
                ts_ = ts[i : i + self.hparams.chunk].detach().clone().to(device)
                self.to(device)
            else:
                rays_ = rays[i : i + self.hparams.chunk]
                ts_ = ts[i : i + self.hparams.chunk]

            rendered_ray_chunks = model.render_rays(
                models=self.models,
                embeddings=self.embeddings,
                rays=rays_,
                ts=ts_,
                N_samples=self.hparams.N_samples,
                perturb=perturb,
                noise_std=noise_std,
                N_importance=self.hparams.N_importance,
                chunk=self.hparams.chunk,
                hp=self.hparams,
                test_time=test_time,
                model_=self,
            )

            for k in list(rendered_ray_chunks):
                if test_time:
                    results[k] += [rendered_ray_chunks[k].cpu()]
                else:
                    results[k] += [rendered_ray_chunks[k]]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)

        self.to(device_prev)

        return results

    def setup(self, stage, reset_dataset=False, train_dataset=None, val_dataset=None):
        kwargs = {"root": self.hparams.root}
        kwargs["vid"] = self.hparams.vid
        kwargs["with_radialdist"] = self.hparams.with_radialdist
        print(f'Dataset ID: {self.hparams.vid}')
        frames_dir = self.hparams.frames_dir

        self.train_dataset = dataset.EPICDiff(
            split=self.hparams.dl_type,
            frames_dir=frames_dir,
            scale=self.hparams.scale,
            cfg=self.hparams,
            model=self,
            **kwargs
        )

        scale = self.hparams.eval.perframe.scale

        if self.val_dataset is None:

            if val_dataset is None:
                self.val_dataset = dataset.EPICDiff(
                    split="val",
                    frames_dir='frames',
                    scale=scale,
                    cfg=self.hparams,
                    **kwargs
                )
            else:
                self.val_dataset = val_dataset

            self.empty_dataset = dataset.rays.EmptyDataset()

            self.maskloader = dataset.annotations.MaskLoader(
                self.val_dataset,
                empty_masks=True
            )

        if self.hparams.model_type == 't-nerf':
            self.embedding_t.set_max_frame_id(max(self.val_dataset.img_ids))


    def configure_optimizers(self):
        eps = 1e-8

        self.optimizer = Adam(
            get_parameters(self.models_to_train),
            lr=self.hparams.lr,
            eps=eps,
            weight_decay=self.hparams.weight_decay,
        )

        lrsched = self.hparams.train.lrsched

        if lrsched == 'cos':
            scheduler = CosineAnnealingLR(
                self.optimizer, T_max=self.hparams.num_epochs, eta_min=eps
            )
        elif lrsched == 'exp':
            gamma = math.exp(math.log(5e-07/self.hparams.lr)/self.hparams.num_epochs)
            scheduler = ExponentialLR(
                self.optimizer,
                gamma=gamma
            )

        if lrsched is None:
            print('--- Training without LR scheduler.')
            return [self.optimizer]
        else:
            return [self.optimizer], [scheduler]

    def train_dataloader(self):
        num_workers = self.hparams.num_workers
        return DataLoader(
            self.train_dataset,
            shuffle=False, # shuffled already in dataset
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.empty_dataset,
            shuffle=False,
            num_workers=0,
            batch_size=1,
            pin_memory=True,
        )


    def training_step(self, batch, batch_nb, disable_logging=False):
        rays, rgbs, ts = batch["rays"], batch["rgbs"], batch["ts"]

        results = self(rays, ts)

        loss_d = self.loss(results, rgbs)

        loss = sum(l for l in loss_d.values())
        with torch.no_grad():
            psnr_ = psnr(results["rgb_fine"], rgbs)

        if not disable_logging:

            self.log("lr", self.optimizer.param_groups[0]["lr"])
            self.log("train/loss", loss)
            for k, v in loss_d.items():
                self.log(f"train/{k}", v, prog_bar=False)
            self.log("train/psnr", psnr_, prog_bar=True)
            if hasattr(self, 'n_pos'):
                self.log("train/npos", self.n_pos, prog_bar=True)

        self.counter += 1

        if self.hparams.data.set.tr.sampled.active:
            if self.counter % self.hparams.data.set.tr.sampled.n_iter == 0:
                self.trainer.reset_train_dataloader(self)

        return loss

    def render(self, sample, t=None, device=None):

        rays, rgbs, ts = (
            sample["rays"],
            sample["rgbs"],
            sample["ts"],
        )

        if t is not None:
            if type(t) is torch.Tensor:
                ts = t.to(rays.device)
            elif type(t) is int:
                ts = torch.ones_like(ts) * t

        rays = rays.squeeze()  # (H*W, 3)
        rgbs = rgbs.squeeze()  # (H*W, 3)
        ts = ts.squeeze()  # (H*W)
        with torch.no_grad():
            results = self(rays, ts, test_time=True)

        return results

    def validation_step(self, batch, batch_nb, is_debug=False):

        idx = sample_validation_frame(self.val_dataset)
        results = evaluate_sample(self.val_dataset, idx, model=self, mask_targ=self.maskloader[idx])
        grid, psnr, avgpre = results['grid'], results['psnr'], results['average_precision']
        grid = torch.FloatTensor(grid).permute(2,0,1)[None,] / 255

        if self.logger is not None:
            self.logger.experiment.add_images(
                "val/GT_pred_depth", grid, self.global_step
            )

        # rgb results have 6 channels for masks, incompatible with GT RGB/loss, remove.
        results = utils.remove_mask_channels(results)

        rgbs = self.val_dataset.rgbs_per_image(idx)

        # for now: set coarse RGB to target, then loss=0, adjust later
        results['rgb_coarse'] = rgbs

        loss_d = self.loss(results, rgbs)
        log = {}

        # to CUDA for compatibility with multi GPU
        log["val_psnr"] = torch.FloatTensor([psnr]).cuda()
        log["val_loss"] = sum(l for l in loss_d.values()).cuda()
        log['val_avgpre'] = torch.FloatTensor([avgpre]).cuda()

        if self.hparams.eval.sampled.active:
            psnr, psnr_static, psnr_nonstatic, average_precision = evaluate_sampled(
                self.sampled_dataset,
                self,
                perc=self.hparams.eval.sampled.perc
            )
            log['val_psnr'] = torch.FloatTensor([psnr]).cuda()
            log['val_psnr_static'] = torch.FloatTensor([psnr_static]).cuda()
            log['val_psnr_nonstatic'] = torch.FloatTensor([psnr_nonstatic]).cuda()
            log['val_avgpre'] = torch.FloatTensor([average_precision]).cuda()

        return log

    def validation_epoch_end(self, outputs):

        # for now excluding multi gpu on validation, hence only single values
        assert len(outputs) == 1
        for k in outputs[0]:
            # 'val_loss', 'val_psnr_static' etc. to 'val/loss', 'val/psnr_static'
            k_ = k.replace('val_', 'val/')
            if k in ['val_psnr', 'val_psnr_static', 'val_avgpre']:
                prog_bar = True
            else:
                prog_bar = False
            self.log(k_, outputs[0][k], prog_bar=prog_bar)


def init_trainer(hparams, logger=None, system=None):
    assert hparams.ckpt_dst is not None
    assert hparams.logs_dst is not None
    checkpoint_callback = pytorch_lightning.callbacks.ModelCheckpoint(
        dirpath=f"{hparams.ckpt_dst}/{hparams.exp_name}",
        filename="{epoch:d}",
        save_top_k=-1,
        )

    logger = pytorch_lightning.loggers.TensorBoardLogger(
        save_dir=hparams.logs_dst,
        name=hparams.exp_name,
        log_graph=False,
    )

    if hparams.train_ratio == -1:
        train_ratio = 20000
    else:
        train_ratio = hparams.train_ratio

    if hparams.ckpt_path is not None:
        ckpt_path = get_latest_ckpt(hparams.ckpt_path)
        hparams.ckpt_path = ckpt_path

    trainer = pytorch_lightning.Trainer(
        max_epochs=hparams.num_epochs,
        callbacks=[checkpoint_callback],
        enable_checkpointing=True,
        resume_from_checkpoint=hparams.ckpt_path,
        logger=logger,
        enable_model_summary=False,
        # progress_bar_refresh_rate=1,
        gpus=hparams.num_gpus,
        num_sanity_val_steps=0,
        val_check_interval=hparams.val_check_interval,
        # val_check_interval=0.25,
        # reload_dataloaders_every_n_epochs=1,
        accelerator=hparams.accelerator,
        benchmark=False,
        limit_train_batches=train_ratio,
        profiler=None,
        precision=16,
    )

    system.trainer = trainer

    return trainer


def main(hparams):
    system = NeuralDiffSystem(hparams)
    trainer = init_trainer(hparams, system=system)
    trainer.fit(system)


if __name__ == "__main__":
    hparams = get_opts()
    main(hparams)
