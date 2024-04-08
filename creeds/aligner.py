
from kartograf.atom_aligner import align_mol_skeletons
from gufe import SmallMoleculeComponent
from rdkit import Chem
from rdkit.Chem import rdMolAlign
from typing import List
import json

def concatenate(ClusterMap : str, clusterSDF_dir: str,  output_dir: str, includeList : List[int] = None):
        
    f = open(ClusterMap, 'r')
    cluster_data = json.load(f)

    #clusters to concatenate
    if includeList is None:
        includeList = list(cluster_data.keys())
    
    for cluster in includeList:
        
        with open(output_dir + "/" + str(cluster) + '.sdf', 'w') as f:
            for item in cluster_data[cluster]:
                with open(clusterSDF_dir + item + ".sdf", 'r') as file:
                    f.write(file.read())

def alignClusters(ClusterMap : str, concatSDF_dir : str, output_dir: str, returnSVG : bool = True):
    '''
    aligns single sdf files
    '''
    # Load the molecules
    clusters = json.load(open(ClusterMap))
    
    for cluster_name in clusters.keys():
        molecules = []

        data = Chem.SDMolSupplier(concatSDF_dir + cluster_name + ".sdf", removeHs=False)

        #first atom is reference
        for i, mol in enumerate(molecules):
            if i == 0:
                ref = mol
                molecules.append(ref)
            else:
                molB = mol
                pyo3a = rdMolAlign.GetO3A(molB, ref)
                pyo3a.Align()
                molecules.append(molB)

        

        #revert back to sdf
        with Chem.SDWriter(output_dir + "aligned/" + cluster_name + ".sdf") as w:    
            for mol in molecules:
                w.write(mol)

        #Visualize
        if returnSVG:
            rd_kit = [mol for mol in molecules]
            SVG = Chem.Draw.MolsToGridImage(rd_kit, useSVG=True, molsPerRow=10)
            with open(output_dir + "visualization/" + cluster_name + ".svg", "w") as f:
                f.write(SVG)

        
if __name__ == '__main__':
    print("Concatenate ligands together...\n")
    
    concatenate('/localhome/lconconi/CREEDS/creeds/output/FFS_cluster04_c/clustersFFS_cluster04_c_MCMS.json',
                '/localhome/lconconi/CREEDS/creeds/output/FFS_cluster04_c/sdf_files/',
                '/localhome/lconconi/CREEDS/creeds/output/FFS_cluster04_c/concatenate/')

    print("Beginning to align Ligands...\n")
    alignClusters('/localhome/lconconi/CREEDS/creeds/output/FFS_cluster04_c/clustersFFS_cluster04_c_MCMS.json',
                 '/localhome/lconconi/CREEDS/creeds/output/FFS_cluster04_c/concatenate/',
                 '/localhome/lconconi/CREEDS/creeds/output/FFS_cluster04_c/', returnSVG=True)
    print("Aligned ligands, they can be found in the output directory" )