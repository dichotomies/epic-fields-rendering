import io
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ..utils import add_title, tqdm, x2im, visualize_depth

def plt_to_im(f, show=False, with_alpha=False, is_notebook=False):
    # f: figure from previous plot (generated with plt.figure())
    buf = io.BytesIO()
    buf.seek(0)
    if is_notebook:
        plt.savefig(buf, format='png', bbox_inches='tight',transparent=True, pad_inches=0)
    else:
        plt.savefig(buf, format="jpg")
    if not show:
        plt.close(f)
    im = Image.open(buf)
    # return without alpha channel (contains only 255 values)
    return np.array(im)[..., : 3 + with_alpha]


def sample_linear(X, n_samples):
    if n_samples == 0:
        n_samples = len(X)
    n_samples = min(len(X), n_samples)
    indices = (np.linspace(0, len(X) - 1, n_samples)).round().astype(np.long)
    return [X[i] for i in indices], indices


def write_summary(results):
    """Log average precision and PSNR score for evaluation."""
    import io
    from contextlib import redirect_stdout

    with io.StringIO() as buf, redirect_stdout(buf):

        n = 0
        keys = sorted(results["metrics"].keys())

        for k in keys:
            print(k.ljust(n), end="\t")
        print()
        for k in keys:
            trail = -2
            lead = 0
            print(f'{results["metrics"][k]:.4f}'[lead:trail].ljust(n), end="\t")
        print()

        output = buf.getvalue()

    return output
