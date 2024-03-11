
from Bio import PDB
import numpy as np
from typing import Optional, Iterable
from collections.abc import Sequence
import torch
from graph import graph_via_threshold
import os
from lib.data.datasets import DeepGraphLibraryIterableDataset
from lib.data.datamodules import BarebonesDataModule
from torch.utils.data import DataLoader


def get_atom_coordinates(
        residue : PDB.Residue,
        ) -> tuple[list, list]:
    """
    Get the atom types and coordinates of a residue.
    """
    atoms = []
    coordinates = []
    for atom in residue:
        atoms.append(atom.get_id()[0])
        coordinates.append(atom.get_coord())
    return atoms, coordinates


atom_embedding_dict = {
    'C' : 0,
    'N' : 1,
    'O' : 2,
    'P' : 3,
    'H' : 4,
}


class RibonucleicAcidConformation(Sequence):
    def __init__(self, path : str) -> None:
        # store the path
        self.path = path

        # load the structure
        parser = PDB.PDBParser()
        structure = parser.get_structure('RNA', path)

        # ensure that there is only one model
        if not len(structure) == 1:
            raise ValueError('The structure must contain only one model.')
        
        # store the single model
        self.model = structure[0]
        self.chain_ids = [chain.get_id() for chain in self.model]


    def get_atom_coordinates(self, chain : int) -> np.ndarray:
        """
        Get the atom coordinates for all residues in a chain.
        """
        atoms = []
        coordinates = []
        for residue in self[chain]:
            if PDB.is_aa(residue):
                continue
            a, c = get_atom_coordinates(residue)
            atoms += a
            coordinates += c
            
        return np.array(atoms), np.array(coordinates)
    

    def get_atom_embeddings(self, chain : int) -> np.ndarray:
        """
        Get the atom embeddings of a chain.
        """
        atoms, coordinates = self.get_atom_coordinates(chain)
        embeddings = np.zeros((len(atoms)))
        for i, atom in enumerate(atoms):
            embeddings[i] =  atom_embedding_dict[atom]
        return embeddings, coordinates
        

    def get_num_chains(self) -> int:
        """
        Return the number of chains in the model.
        """
        return len(self.model)
    

    def __len__(self) -> int:
        """
        Return the number of residues in all chains of the model.
        """
        return sum([len(chain) for chain in self.model])
    

    def __getitem__(self, index : int) -> PDB.Residue:
        """
        Return the chain of the model at the given index.
        """
        return self.model[self.chain_ids[index]]



def load_graphs_from_pdb(dir : str, threshold : float):
    """
    Load RNA conformations from a directory of PDB files, and return a list of
    DGL graphs representing the conformations.
    """

    # get the files from the directory
    files = [
        os.path.join(dir, file) 
        for file in os.listdir(dir) 
        if file.endswith('.pdb')
        ]
    
    # load the RNA conformations
    graphs = []
    for file in files:

        # load the RNA conformation
        rna = RibonucleicAcidConformation(file)

        # only use conformations with a single chain
        if rna.get_num_chains() > 1:
            continue

        # get the atom embeddings of the lone chain
        atom, coordinates = rna.get_atom_embeddings(0)
        
        # construct the graph
        graph = graph_via_threshold(torch.tensor(coordinates), threshold)
        graph.ndata['atoms'] = torch.tensor(atom, dtype=torch.int32)
        graphs.append(graph)

    return graphs



class RibonucleicAcidDataModule(BarebonesDataModule):
    """
    A DataModule for netCDF data, providing functionality for loading, 
    transforming, and batching the data.

    Parameters:
    ----------

    """
    def __init__(
            self, 
            threshold : float,
            train_dir : Optional[str] = None,
            validate_dir : Optional[str] = None,
            test_dir : Optional[str] = None,
            *args, **kwargs,
            ) -> None:
        super().__init__(*args, **kwargs)

        # store the arguments
        self.threshold = threshold
        self.directories = {
            'train' : train_dir,
            'validate' : validate_dir,
            'test' : test_dir,
            }

        # raise an error if the number of workers is greater than 1
        if self.num_workers > 1:
            raise ValueError(
                'The number of workers cannot exceed 1 for netCDF datasets.' \
                ' Exactly one is preferable.'
                )


    def create_datasets(
            self, 
            phase: str, 
            rank: int, 
            world_size: int,
            ) -> Iterable:
        """
        Create a dataset for the specified phase, if a path to the data is
        specified.
        """
        if self.directories[phase] is not None:
            # load the graphs from the directory
            graphs = load_graphs_from_pdb(self.directories[phase], self.threshold)

            # construct the dataset
            return DeepGraphLibraryIterableDataset(
                graphs=graphs, 
                batch_size=self.batch_size,
                rank=rank,
                world_size=world_size,
                )
    

    def create_dataloaders(self, phase: str) -> DataLoader:
        """
        Create a dataloader for the specified phase.

        Parameters:
        ----------
        phase (str): 
            The phase for which to create the dataloaders. Can be one of 
            'train', 'val', 'test', or 'predict'.

        Returns:
        -------
        torch.utils.data.DataLoader: 
            The dataloader for the specified phase.
        """        
        if phase not in ['train', 'validate', 'test', 'predict']:
            raise ValueError(
                f'Unknown phase {phase}. Please specify one of "train", "val", "test", or "predict".'
                )

        if self.data[phase] is not None:
            if phase == 'train':
                return DataLoader(
                    dataset = self.data[phase],
                    num_workers = self.num_workers,
                    batch_size = (None if self.num_workers <= 1 else self.num_workers),
                    multiprocessing_context = 'fork' if torch.backends.mps.is_available() and self.num_workers > 0 else None,
                    )
            else:
                return [DataLoader(
                    dataset = data,
                    num_workers = self.num_workers,
                    batch_size = (None if self.num_workers <= 1 else self.num_workers),
                    multiprocessing_context = 'fork' if torch.backends.mps.is_available() and self.num_workers > 0 else None,
                ) for data in self.data[phase]]