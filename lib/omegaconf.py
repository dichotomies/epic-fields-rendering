
from omegaconf import OmegaConf

def conf_to_list(d, l=None, parent_key=''):
    if parent_key == '':
        d = OmegaConf.to_container(d)
        l = []
    for k in d:
        expanded_key = parent_key + '.' + k
        l.append(expanded_key)
        if type(d[k]) == dict:
            conf_to_list(d[k], l, parent_key=expanded_key)
    return l


def conf_symmetric_difference(conf1, conf2):
    keys = set(conf_to_list(conf1)).symmetric_difference(conf_to_list(conf2))
    return keys