
import torch
import torch.nn as nn
import dgl 
from se3_transformer.model import SE3Transformer
from se3_transformer.model.fiber import Fiber

def sinusoidal_embedding(seq_len : int, embedding_dim : int) -> torch.Tensor:
    """
    Return the sinusoidal embedding of a sequence of length seq_len and
    dimension embedding_dim.

    Parameters:
    -----------
    seq_len : int
        The length of the sequence.
    embedding_dim : int
        The dimension of the embedding.

    Returns:
    --------
    torch.Tensor
        A tensor of shape (seq_len, embedding_dim) representing the sinusoidal
        embedding of the sequence.
    """
    # create the positions
    positions = torch.arange(seq_len, dtype=torch.float32)

    # create the dimensions
    dimensions = torch.arange(embedding_dim, dtype=torch.float32)

    # calculate the angles
    angles = positions.unsqueeze(-1) / (10000 ** (2 * (dimensions // 2) / embedding_dim))

    # calculate the sinusoidal embedding
    sinusoidal_embedding = torch.zeros(seq_len, embedding_dim)
    sinusoidal_embedding[:, 0::2] = torch.sin(angles[:, 0::2])
    sinusoidal_embedding[:, 1::2] = torch.cos(angles[:, 1::2])

    return sinusoidal_embedding



class RibonucleicAcidSE3Transformer(nn.Module):
    def __init__(
            self, 
            atom_embedding_dim : int,
            timestep_embedding_dim : int,
            num_timesteps : int,
            hidden_size : int,
            num_layers : int,
            num_heads : int,
            num_atom_types : int,
            ) -> None:
        super().__init__()

        # ensure that the hidden size is divisible by the number of heads
        if not hidden_size % num_heads == 0:
            raise ValueError(
                f'The hidden size {hidden_size} must be divisible by the number of heads {num_heads}.'
                )
        
        # construct the timestep embedding
        self.register_buffer(
            name='timestep_embedding', 
            tensor=sinusoidal_embedding(num_timesteps, timestep_embedding_dim),
            )
        
        # construct the fibers
        fiber_in = Fiber(
            {
                '0' : atom_embedding_dim + timestep_embedding_dim,
                '1' : 1,
            }
        )
        fiber_hidden = Fiber.create(3, hidden_size)
        fiber_out = Fiber.create(2, 1)
        
        # construct the SE(3)-Transformer model
        self.model = SE3Transformer(
            num_layers=num_layers,
            fiber_in=fiber_in,
            fiber_hidden=fiber_hidden,
            fiber_out=fiber_out,
            num_heads=num_heads,
            channels_div=1,
        )

        # construct the atom embedding layer
        self.atom_embedding = nn.Embedding(num_atom_types, atom_embedding_dim)


    def forward(self, graph : dgl.DGLGraph, t : int) -> torch.Tensor:
        """
        Pass the atomic graph through the SE(3)-Transformer model, and update
        the node features with the output of the model.

        The graph must have the following node features:
        - atoms : torch.Tensor
            A tensor of shape (N, 1) representing the type of each atom.
        - coordinates : torch.Tensor
            A tensor of shape (N, 3) representing the coordinates of each atom.

        Parameters:
        -----------
        graph : dgl.DGLGraph
            The atomic graph.
        t : int
            The current timestep of the diffusion process.

        Returns:
        --------
        dgl.DGLGraph
            The atomic graph with updated node features.
        """

        # copy the graph
        graph = graph.local_var()

        # get the atom embeddings
        atom_types = graph.ndata['atoms'].unsqueeze(1)
        atom_embeddings = self.atom_embedding(atom_types).permute(0, 2, 1)

        # get the timestep embeddings
        timestep_embeddings = self.timestep_embedding[t].unsqueeze(0).expand(atom_embeddings.shape[0], -1, -1).permute(0, 2, 1)

        # concatenate the atom and timestep embeddings
        atom_embeddings = torch.cat([atom_embeddings, timestep_embeddings], dim=1)

        # centre the coordinates
        atom_coordinates = graph.ndata['coordinates'].unsqueeze(1)
        atom_coordinates -= atom_coordinates.mean(dim=0, keepdim=True)

        # pass the atom embeddings through the SE(3)-Transformer model
        _, coordinates = self.model(graph, {'0': atom_embeddings, '1' : atom_coordinates}, {}).values()

        # update the node features
        graph.ndata['coordinates'] = coordinates

        return graph