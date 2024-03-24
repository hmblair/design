
import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from lib.training.modules import DenoisingDiffusionModule
from model import RibonucleicAcidSE3Transformer

ckpt = 'checkpoints/diff.ckpt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = DenoisingDiffusionModule(
    model=RibonucleicAcidSE3Transformer(
        atom_embedding_dim=26,
        timestep_embedding_dim=6,
        num_timesteps=1000,
        num_layers=8,
        num_heads=8,
        num_atom_types=4,
        k_nearest_neighbors=25,
    ),
    objective=nn.MSELoss(),
    diffused_variable='coordinate',
)
model = model.to(device)
model.eval()

# load the checkpoint
state_dict = torch.load(ckpt, map_location=device)['state_dict']
model.load_state_dict(state_dict)

# generate a sample
shape = (1, 50, 3)
sequence = torch.randint(0, 4, (1, 50))

out = model(shape, sequence=sequence)
out = out.detach().numpy()

# save the output to an xarray dataset
import xarray as xr
data = xr.Dataset(
    {
        'coordinate' : (['residue', 'xyz'], out),
        'sequence' : (['residue'], sequence.squeeze().numpy())
    },
)