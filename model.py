
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
            k_nearest_neighbors : int,
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

        # store the number of nearest neighbors
        self.k_nearest_neighbors = k_nearest_neighbors


    def forward(
            self, 
            sequence : torch.Tensor, 
            coordinate : torch.Tensor, 
            t : int,
            ) -> torch.Tensor:
        """
        Using the given coordinates, a geometric graph is constructed and passed
        through the SE(3)-Transformer model. The updated coordinates are returned.
        The timestep is used to condition the model for the reverse diffusion
        process.
        """
        # get the batch size
        b, n, _ = coordinate.shape

        # get the atom embeddings
        sequence = sequence.reshape(-1, 1)
        atom_embeddings = self.atom_embedding(sequence).permute(0, 2, 1)

        # get the timestep embeddings
        timestep_embeddings = self.timestep_embedding[t].unsqueeze(0).expand(atom_embeddings.shape[0], -1, -1).permute(0, 2, 1)

        # concatenate the atom and timestep embeddings
        atom_embeddings = torch.cat([atom_embeddings, timestep_embeddings], dim=1)

        # centre the coordinates
        coordinate -= coordinate.mean(dim=1, keepdim=True)
        coordinate = coordinate

        # construct the graph
        graph = dgl.knn_graph(coordinate, self.k_nearest_neighbors)
        reshaped_coordinates = coordinate.view(-1, 3)
        graph.edata['rel_pos'] = reshaped_coordinates[graph.edges()[0]] - reshaped_coordinates[graph.edges()[1]]

        # pass the graph through the SE(3)-Transformer model
        atoms, coordinate = self.model(
            graph, {'0' : atom_embeddings, '1' : reshaped_coordinates.unsqueeze(1)},
        ).values()

        return coordinate.squeeze(1).reshape(b, n, 3)