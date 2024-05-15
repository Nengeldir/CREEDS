#!/usr/bin/env python
# coding: utf-8

# In[6]:


# This notebook uses the (not yet merged) expanding_for_multistates branch of kartograf
# https://github.com/OpenFreeEnergy/kartograf/tree/expanding_for_multistates

from kartograf import KartografAtomMapper, SmallMoleculeComponent
from kartograf.atom_mapper import mapping_algorithm
from rdkit import Chem
import numpy as np
mapper = KartografAtomMapper()
data = Chem.SDMolSupplier('Cluster_2.sdf', removeHs=False)

molecules = []
ref = None
for i, mol in enumerate(data):
    if i == 6:
        ref = mol
        

for i, mol in enumerate(data):
    if i == 6:
        molecules.append(mol)
    else:
        molB = mol
        pyo3a = Chem.rdMolAlign.GetO3A(molB, ref)
        pyo3a.Align()
        molecules.append(molB)

components = [SmallMoleculeComponent(mol) for mol in molecules]

mapping = mapper.suggest_multistate_mapping(components, greedy=False, map_hydrogens=False, max_d = 1)

import json

with open('mapping.json', 'w') as f:
    json.dump(mapping, f)


# In[7]:


from kartograf.utils.multistate_visualization import visualize_multistate_mappings_2D

visualize_multistate_mappings_2D(components, mapping)


# In[ ]:




