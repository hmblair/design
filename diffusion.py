 
import torch
import torch.nn as nn

from lib.training.modules import DenoisingDiffusionModule
from lib.models.recurrent import RecurrentDecoder



def get_timestep_encoding(max_timesteps : int, encoding_dim : int) -> torch.Tensor:

    encoding = torch.zeros(max_timesteps, encoding_dim)
    pos = torch.arange(max_timesteps)
    for i in range(encoding_dim):
        if i % 2 == 0:
            encoding[:, i] = torch.sin(pos / 10000 ** (2 * i / encoding_dim))
        else:
            encoding[:, i] = torch.cos(pos / 10000 ** (2 * i / encoding_dim))

    return encoding




class RecurrentDiffusion(RecurrentDecoder):
    def __init__(
            self, 
            max_timesteps : int, 
            timestep_encoding_dim : int, 
            in_size : int,
            *args, **kwargs, 
            ) -> None:
        super().__init__(in_size = in_size + timestep_encoding_dim, *args, **kwargs)
        self.timestep_encoding = get_timestep_encoding(max_timesteps, timestep_encoding_dim)


    def forward(self, x : torch.Tensor, t : int) -> torch.Tensor:
        t = torch.tensor(t, device = x.device)
        t = self.timestep_encoding[t].repeat(x.size(0), x.size(1), 1)
        x = torch.cat([x, t], dim = -1)
        return super().forward(x)


# create a model
model = RecurrentDiffusion(
    in_size = 3,
    hidden_size = 16, 
    out_size = 3,
    num_layers = 1,
    max_timesteps = 1000,
    timestep_encoding_dim = 4,
)

# create some data
num_datapoints = 10
seq_len = 5
data = torch.randn(num_datapoints, seq_len, 3)

# create a diffusion module
diffusion_module = DenoisingDiffusionModule(
    model = model, 
    beta = torch.linspace(0.02, 0.001, 1000)
)

# take a sample
sample = diffusion_module(torch.Size([1, seq_len, 3]))

breakpoint()