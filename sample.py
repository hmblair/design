
import torch
import torch.nn as nn
import xarray as xr

from lib.training.modules import DenoisingDiffusionModule
from model import RibonucleicAcidSE3Transformer

ckpt = 'checkpoints/diff.ckpt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = DenoisingDiffusionModule(
    model=RibonucleicAcidSE3Transformer(
        atom_embedding_dim=12,
        atom_embedding_layers=2,
        timestep_embedding_dim=4,
        num_timesteps=1000,
        num_layers=6,
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

# get a sequence from the train dataset
file = 'data/train/76.nc'
sequence = xr.open_dataset(file)['sequence'].values[0]
shape = (1,) + sequence.shape + (3,)

# convert the sequence to a tensor, add a batch dimension, and move to the device
sequence = torch.tensor(sequence).unsqueeze(0).to(device)

out = model(shape, sequence=sequence)
out = out.detach().cpu().numpy()

# save the output to an xarray dataset
import xarray as xr
data = xr.Dataset(
    {
        'coordinate' : (['residue', 'xyz'], out.squeeze(0)),
        'sequence' : (['residue'], sequence.squeeze().cpu().numpy())
    },
)
data.to_netcdf('sample.nc')
