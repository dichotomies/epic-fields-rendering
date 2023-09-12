
import omegaconf
from omegaconf import OmegaConf
from omegaconf import DictConfig, OmegaConf

import math

import os
from hydra import initialize, initialize_config_module, initialize_config_dir, compose


def get_opts(**kwargs):

    OmegaConf.register_new_resolver("float", lambda x: float(x), replace=True)
    OmegaConf.register_new_resolver("mul", lambda x, y: x * y, replace=True)
    OmegaConf.register_new_resolver("sqrt", lambda x: math.sqrt(float(x)), replace=True)

    # either notebook or CLI
    if len(kwargs) > 0:
        cfg_update = DictConfig(kwargs)
    else:
        cfg_update = OmegaConf.from_cli()

    with initialize(config_path="conf"):
        cfg = compose('default')

    # make config editable (e.g. via merge)
    OmegaConf.set_struct(cfg, False)

    # merge with args provided in notebook or through CLI
    cfg = omegaconf.OmegaConf.merge(cfg, cfg_update)

    OmegaConf.resolve(cfg)

    return cfg
