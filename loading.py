
from Bio import PDB
import numpy as np
from typing import Optional
from collections.abc import Sequence
import torch
from graph import graph_via_threshold
import os
from lib.data.datasets import DeepGraphLibraryIterableDataset

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


import pytorch_lightning as pl
from torch.utils.data import DataLoader

class RibonucleicAcidDataModule(pl.LightningDataModule):
    def __init__(self, dir : str, batch_size : int, threshold : float):
        super().__init__()
        self.dir = dir
        self.batch_size = batch_size
        self.threshold = threshold
        self.graphs = None


    def setup(self, stage : Optional[str] = None):
        graphs = load_graphs_from_pdb(self.dir, self.threshold)
        self.graphs = DeepGraphLibraryIterableDataset(
            graphs, batch_size=self.batch_size,
            )


    def train_dataloader(self):
        return DataLoader(self.graphs, batch_size=None)


    def val_dataloader(self):
        return DataLoader(self.graphs, batch_size=None)


    def test_dataloader(self):
        return DataLoader(self.graphs, batch_size=None)