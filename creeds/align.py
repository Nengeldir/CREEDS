
from kartograf.atom_aligner import align_mol_skeletons
from gufe import SmallMoleculeComponent
from rdkit import Chem
from rdkit.Chem import rdMolAlign

import json

def alignCluster(ClusterMap : str, clusterSDF_dir : str, returnSVG : bool = True):
    # Load the molecules
    clusters = json.load(open(ClusterMap)).keys()

    for cluster in clusters:
        data = Chem.SDMolSupplier(clusterSDF_dir + "/" + cluster + ".sdf", removeHs=False)

        #first atom is reference
        molecules = []

        for i, mol in enumerate(data):
            if i == 0:
                ref = mol
                molecules.append(ref)
            else:
                molB = mol
                pyo3a = rdMolAlign.GetO3A(molB, ref)
                pyo3a.Align()
                molecules.append(molB)

        

        #revert back to sdf
        with Chem.SDWriter("output/aligned/" + cluster + ".sdf") as w:    
            for mol in molecules:
                w.write(mol)

        #Visualize
        if returnSVG:
            rd_kit = [mol for mol in molecules]
            SVG = Chem.Draw.MolsToGridImage(rd_kit, useSVG=True, molsPerRow=10)
            with open("output/visualization/" + cluster + ".svg", "w") as f:
                f.write(SVG)

        
if __name__ == '__main__':
    alignCluster('clusters.json', 'output', returnSVG=True)