#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This notebook uses the (not yet merged) expanding_for_multistates branch of kartograf
# https://github.com/OpenFreeEnergy/kartograf/tree/expanding_for_multistates

from kartograf import KartografAtomMapper, SmallMoleculeComponent
from rdkit import Chem

mapper = KartografAtomMapper()
for i in range(10):
    print(i)
    ligands = [mol for mol in Chem.SDMolSupplier('Cluster_' + str(i) +'.sdf', removeHs=False)]
    components = [SmallMoleculeComponent(mol) for mol in ligands]

    mapping = mapper.suggest_multistate_mapping(components, greedy=False, map_hydrogens=False)

    import json

    with open('mapping' + str(i) + '.json', 'w') as f:
        json.dump(mapping, f)


# In[15]:


from kartograf.utils.multistate_visualization import visualize_multistate_mappings_2D

visualize_multistate_mappings_2D(components, mapping)


# In[ ]:




