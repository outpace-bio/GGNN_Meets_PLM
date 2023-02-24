# PPI: Protein-Protein Interfaces


## Overview

This task relates to predicting which pairs of amino acids, spanning two
different proteins, will interact upon binding (when they form a complex).

Amino acids are defined as interacting if any of their heavy atoms are within 6
Angstroms from one another.  


## Datasets

- raw: 
   - DB5: the complexes present in DB5 (see add. inf.)
   - DIPS: pairs of chains from DIPS (see add. inf.)
- splits:
   - DIPS-split: DIPS dataset, split by sequence identity (see add. inf.)


## Usage

from atom3d.datasets import LMDBDataset
dataset = LMDBDataset(PATH_TO_LMDB)
print(len(dataset))  # Print length
print(dataset[0])  # Print 1st entry


## Format

Each entry in the dataset contains the following keys:

['atoms_pairs'] (pandas.DataFrame) Atom coordinates of the complexes.
['atoms_neighbors'] (pandas.DataFrame) Indicates which amino acids interacting in each complex.
['id'] (str) Contains the PDB filenames, model numbers, and chain IDs of the complexes
    (format: <PDB_FILE>_<MODEL_NUM1>_<CHAIN_ID1>_<PDB_FILE>_<MODEL_NUM2>_<CHAIN_ID2>)  
['types'] (dict) Type of each entry.
['file_path'] (str) Path to the LMDB.


## Additional Information

The ensemble consists of a given protein complex, with subunits corresponding 
to the two individual proteins involved in the complex.

### DB5 Dataset

This dataset consists of the complexes present in DB5 (Vreven et al., 2015).
Specifically, these are selected complexes where each protein was experimentally
determined on its own, as well as bound to its partner.  There are 230 in total.
This is used as a gold standard test set.

### DIPS Dataset

This dataset is derived from DIPS (Townshend et al., 2019).  It consists of all
pairs of chains in the same file in the PDB that pass certain criteria:

- Molecule type protein
- At least 50 amino acids.
- Experimental resolution better than 3.5 Angstroms.
- Experimental method used to determine either Cryo-EM or X-Ray Crystallography.
- Buried surface area upon binding of more than 500 Angstroms.
- Less than 30% sequence identity to any protein chain in DB5
- Pair of proteins do not match SCOP superfamily of any pair of proteins in DB5.

There are 134215 pairs of chains total left after pruning.

### DIPS-Split Dataset

This dataset has the same data as the DIPS dataset, but is split
by sequence identity, so that no chain in any set overlaps more than 30% with
any chain in other sets.

Split is approximately 80/10/10.

