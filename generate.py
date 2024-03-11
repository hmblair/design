
import numpy as np

residue_to_atom_dict = {
    'A' : ['N'] * 5 + ['C'] * 5 + ['H'] * 5 + ['O'] * 0,
    'C' : ['N'] * 3 + ['C'] * 4 + ['H'] * 5 + ['O'] * 1,
    'G' : ['N'] * 5 + ['C'] * 5 + ['H'] * 5 + ['O'] * 1,
    'U' : ['N'] * 2 + ['C'] * 4 + ['H'] * 4 + ['O'] * 2,
}

sugar_backbone = ['C'] * 5 + ['H'] * 10 + ['O'] * 9 + ['P'] * 1

def get_atoms_with_random_coordinates(sequence : str) -> np.ndarray:
    """
    Take in an RNA sequence, and return a numpy array of atoms with random
    coordinates, along with an array of the atom types.
    """
    atoms = []
    for residue in sequence:
        atoms += residue_to_atom_dict[residue] + sugar_backbone
    
    # generate random coordinates
    coordinates = np.random.randn(len(atoms), 3)

    return np.array(atoms), coordinates


def to_pdb(atoms: np.ndarray, coordinates: np.ndarray, filename: str):
    with open(filename, 'w') as file:
        for i, (atom, coord) in enumerate(zip(atoms, coordinates), start=1):
            # Assuming atoms array includes atom names and possibly residue info
            # Format: ATOM serial name altLoc resName chainID resSeq x y z occupancy tempFactor element
            # Simplified version below; adjust format as needed for your specific data
            file.write(f"ATOM  {i:5d} {atom:<4} RES A   1    {coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00  0.00           \n")



sequence = 'GCGCGCAAU'
atoms, coordinates = get_atoms_with_random_coordinates(sequence)

to_pdb(atoms, coordinates, 'rna.pdb')