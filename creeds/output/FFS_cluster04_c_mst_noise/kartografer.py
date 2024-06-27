#!/usr/bin/env python
# coding: utf-8

# ### !!! This notebook uses the (not yet merged) expanding_for_multistates branch of kartograf !!!
# ### https://github.com/OpenFreeEnergy/kartograf/tree/expanding_for_multistates

# # Dual Topology restraints mapping

# ### Adding restraints based on multistate MCS (works in simple cases)

# In[3]:


from kartograf import KartografAtomMapper, SmallMoleculeComponent
from rdkit import Chem
import json

mapper = KartografAtomMapper()

molecules = [mol for mol in Chem.SDMolSupplier('Cluster_9.sdf', removeHs=False)]
components = [SmallMoleculeComponent(mol) for mol in molecules]

mapping = mapper.suggest_multistate_mapping(components, greedy=False, map_hydrogens=False)

with open('mapping.json', 'w') as f:
    json.dump(mapping, f)


# In[4]:


from kartograf.utils.multistate_visualization import visualize_multistate_mappings_2D

visualize_multistate_mappings_2D(components, mapping)

