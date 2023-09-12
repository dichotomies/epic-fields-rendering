import torch
from torch import nn


class Loss(nn.Module):
    """
    Name abbreviations:
        c_l: coarse color loss
        f_l: fine color loss
        b_l: beta loss
        s_l: sigma loss
    """

    def __init__(self, lambda_u=0.01, model_type='neuraldiff'):
        """
        lambda_u: regularisation for sigmas.
        """
        super().__init__()
        self.lambda_u = lambda_u
        self.model_type = model_type

    def forward(self, inputs, targets):
        ret = {}
        ret["c_l"] = 0.5 * ((inputs["rgb_coarse"] - targets) ** 2).mean()
        if "rgb_fine" in inputs:
            ret["f_l"] = (
                (inputs["rgb_fine"] - targets) ** 2
                / (2 * inputs["beta"].unsqueeze(1) ** 2)
            ).mean()
            ret["b_l"] = torch.log(inputs["beta"]).mean()
            ret["s_l"] = self.lambda_u * inputs["transient_sigmas"].mean()
            if self.model_type == 'neuraldiff':
                ret["s_l"] = ret["s_l"] + self.lambda_u * inputs["person_sigmas"].mean()

        return ret

def calc_transient_reg(batch, results, trp_mask):
    label_mask = (batch['label'] > 0).flatten(0, 1).squeeze()
    transient_weights_sum_trp = results['transient_weights_sum'][trp_mask][label_mask]
    l = ((1 - transient_weights_sum_trp) ** 2).mean()
    return l