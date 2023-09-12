import torch


class PosEmbedding(torch.nn.Module):
    def __init__(self, max_logscale, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        """
        super().__init__()
        self.funcs = [torch.sin, torch.cos]

        if logscale:
            self.freqs = 2 ** torch.linspace(0, max_logscale, N_freqs)
        else:
            self.freqs = torch.linspace(1, 2 ** max_logscale, N_freqs)

    def forward(self, x):
        out = [x]
        for freq in self.freqs:
            for func in self.funcs:
                out += [func(freq * x)]

        return torch.cat(out, -1)


class LatentEmbedding:
    def __init__(self):
        pass


class TimeEmbedding(torch.nn.Module):
    def __init__(self, max_logscale, N_freqs, logscale=True):
        super().__init__()
        self.embedding = PosEmbedding(max_logscale, N_freqs, logscale)
        self.max_frame_id = None

    def set_max_frame_id(self, max_frame_id):
        print('setting max frame for time embedding.')
        self.max_frame_id = max_frame_id

    def forward(self, x):
        # to range [-1, +1]
        x = (((x / self.max_frame_id) - 0.5) * 2)
        x = x.view(-1, 1)
        return self.embedding(x)


class LREEmbedding(torch.nn.Module):
    """
    As desribed in "Smooth Dynamics", low rank expansion of trajectory states.
    """

    def __init__(self, N=1000, D=16, K=21):
        super().__init__()
        self.embedding = PosEmbedding(K // 2 - 1, K // 2)
        self.W = W = torch.nn.Parameter(torch.FloatTensor(K, D))
        torch.nn.init.kaiming_uniform_(W)
        # normalise input range to [-1, +1]
        self.input_range = (torch.arange(0, N).view(-1, 1) / (N - 1) - 1 / 2) * 2
        self.register_buffer('P', self.embedding(self.input_range))

    def __call__(self, indices):
        L = self.P @ self.W
        return L[indices]
