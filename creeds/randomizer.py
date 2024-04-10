from random import sample
from typing import List, Optional
import json
from kartograf.atom_aligner import align_mol_skeletons
from gufe import SmallMoleculeComponent
from rdkit import Chem
import os

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

def concatenate_all(clusterSDF_dir: str, output_file:str):
    '''
    Concatenate all the sdf files in the clusterSDF_dir into a single file
    '''

    with open(output_file + ".sdf", 'w') as f:
        for root, dirs, files in os.walk(clusterSDF_dir):
            for file in files:
                if file.endswith(".sdf"):
                    with open(os.path.join(root, file), 'r') as file:
                        f.write(file.read())

def alignClusters(ClusterMap : str, concatSDF_dir : str, output_dir: str, returnSVG : bool = True):
    '''
    aligns sdf files into one single sdf file based on a clusterMap
    '''
    # Load the molecules
    clusters = json.load(open(ClusterMap))
    
    for cluster_name in clusters.keys():
        molecules = []

        data = Chem.SDMolSupplier(concatSDF_dir + cluster_name + ".sdf", removeHs=False)

        #first atom is reference
        for i, mol in enumerate(data):
            if i == 0:
                ref = mol
                molecules.append(ref)
            else:
                molB = mol
                pyo3a = Chem.rdMolAlign.GetO3A(molB, ref)
                pyo3a.Align()
                molecules.append(molB)

        
        print(molecules)
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

def randomize_cluster_based(cluster_map : str, 
                            sdf_files : str, 
                            output_dir: str, 
                            simulation_clsuter_map = "simClusterMap.json",
                            max_num_simulations: int = 10,
                            fix_simulation_num : bool = True,
                            fix_ligand_num : bool = True,
                            numLigands_per_sim : int = 15,
                            returnSVG = True
                            ):
    '''
    takes in a cluster_map, returns a partition with overlap to simulate in REEDS

    parameters
        cluster_file
        output_dir
        randomize_Cluster
    '''
    # create random overlapping sets of numLigands
    # num Ligands is meant without the overlap

    cluster_json = json.load(open(cluster_map))
    
    # check how many clusters were generated
    # if there are too many Ligands in one cluster depending on the mode split them randomly or throw an exception to the user
    # if there are too many Clusters based on the mode clump the fewest ones together or throw an exception to the user

    
    if len(cluster_json.keys()) <= max_num_simulations:
        if fix_simulation_num:
            pass # TODO:
        else:
            print("The number of clusters is more than the specified number of simulations, \
                  please fix the clustering or change the number of simulations")
            return -1
    
    simulation_ligands = {}

    for cluster_name in cluster_json.keys():
        numMolecules = len(cluster_json[cluster_name])

        if numLigands_per_sim < numMolecules:
            if fix_ligand_num:
                pass # TODO:
            else:
                print("The cluster ", cluster_name, " has too many ligands in regards to the number of \
                      maximum allowed ligands in a REEDS simulation. Either fix the clustering map, set \
                      the fix_ligand_num flag to true, or increase the number of maximum number of \
                      ligands in a simulation.")
                return -1
        else:
            simulation_ligands[cluster_name] = cluster_json[cluster_name]

    num_simulations = len(simulation_ligands.keys())

    # Make the simulation liands overlapping i.e. add the next ligand to the set
    simulation_keys = list(simulation_ligands.keys())
    for i, cluster in enumerate(simulation_keys):
        next_key = (i + 1) % num_simulations
        next_cluster_name = simulation_keys[next_key]
        simulation_ligands[cluster].append(simulation_ligands[next_cluster_name][0])
    
    #write simulation cluster map to json and save it to file
    
    with open(output_dir + "/sim_map.json", "w") as json_out:
        json_sim_map = json.dumps(simulation_ligands)
        json_out.write(json_sim_map)

    #concatenate single sdf files into the specific clustered sdf files

    for cluster_name in simulation_ligands.keys():
        with open(output_dir + cluster_name + ".sdf", "w") as sim_sdf:
            for ligand in simulation_ligands[cluster_name]:
                lig = open(sdf_files + ligand + ".sdf", "r")
                sim_sdf.write(lig.read())
    

    # align each simulation file 
    for cluster_name in simulation_ligands.keys():
        molecules = []

        data = Chem.SDMolSupplier(output_dir + cluster_name + ".sdf", removeHs=False)

        #first atom is reference
        for i, mol in enumerate(data):
            if i == 0:
                ref = mol
                molecules.append(ref)
            else:
                molB = mol
                pyo3a = Chem.rdMolAlign.GetO3A(molB, ref)
                pyo3a.Align()
                molecules.append(molB)

        #revert back to sdf
        with Chem.SDWriter(output_dir + cluster_name + ".sdf") as w:
            for mol in molecules:
                w.write(mol)

        #Visualize
        if returnSVG:
            rd_kit = [mol for mol in molecules]
            SVG = Chem.Draw.MolsToGridImage(rd_kit, useSVG=True, molsPerRow=10)
            with open(output_dir + "visualization/" + cluster_name + ".svg", "w") as f:
                f.write(SVG)

  
def randomize(output_dir : str,
              sdf_files : str,
              ligand_list : str = None, 
              cluster_map_file : str = None, 
              cluster_name : str = None, 
              max_num_Ligands: int = None, 
              max_num_Simulations: int = None,
              returnSVG : bool = True):
    '''
    ligand_list is a list of all ligands to be used in a simulation
    '''

    # ERRORS Sanity Check
    # ------------------------
    # no max_num_Ligands (or smaller equal to 0) and no num_simulations (or smaller equal to 0) is given 
    if (max_num_Ligands is None or max_num_Ligands <= 0) and (max_num_Simulations is None or max_num_Simulations <= 0):
        print("Please provide either max_num_Ligands or num_Simulations.")
        print("Furthermore check that max_num_Ligands and or num_Simulation is positive and greater one.")
        return -1

    # no ligand_list and no cluster_map_file is given
    if ligand_list is None and cluster_map_file is None:
        print("Either enter a list of ligands to be prepared to the REEDS simulation or enter the path to \
              a cluster_map with the specific cluster that should be prepared.")
    
    # no list of ligands is given but a cluster_map without a specific cluster_name 
    if (ligand_list is None) and (cluster_map_file is not None) and (cluster_name is None):
        print("Please enter a specific cluster_name to be prepared for the REEDS simulation, \
              this function only takes a single cluster in.")
    
    #Errors check out, determine the ligand_list
    if ligand_list is None:
        f = open(cluster_map_file)
        cluster_map = json.load(f)
        ligand_list = cluster_map[cluster_name]

    # calculate the number of simulations
    if max_num_Simulations is None:
        max_num_Simulations = (len(ligand_list) + max_num_Ligands - 1) // max_num_Ligands
    else: # num_simulations is given
        max_num_Ligands = (len(ligand_list) + max_num_Simulations - 1) // max_num_Simulations

    sim_files = {}
    # Choose a random sample of size max_num_Ligands
    for sim in range(max_num_Simulations):
        if len(ligand_list) < max_num_Ligands:
            max_num_Ligands = len(ligand_list)
        sim_files[sim] = sample(ligand_list, max_num_Ligands)
        #remove sample from ligand_list
        ligand_list = [x for x in ligand_list if x not in sim_files[sim]]
        
    
    # Now append the first ligand to the previous subset
    sim_keys = list(sim_files.keys())
    for i, sim in enumerate(sim_keys):
        next_key = sim_keys[(i + 1) % len(sim_keys)]
        sim_files[sim].append(sim_files[next_key][0])

    # Write files out into separate sdf files
    for sim in sim_keys:
        simulation_file = open(output_dir + "simulation_" + str(sim) + ".sdf", "w")
        for ligand in sim_files[sim]:
            lig = open(sdf_files + "/" + ligand + ".sdf", "r")
            simulation_file.write(lig.read())

    # Align the simulation files
    for sim in sim_keys:
        molecules = []

        data = Chem.SDMolSupplier(output_dir + "simulation_" + str(sim) + ".sdf", removeHs=False)

        #first atom is reference
        for i, mol in enumerate(data):
            if i == 0:
                ref = mol
                molecules.append(ref)
            else:
                molB = mol
                pyo3a = Chem.rdMolAlign.GetO3A(molB, ref)
                pyo3a.Align()
                molecules.append(molB)

        #revert back to sdf
        with Chem.SDWriter(output_dir + "simulation_" + str(sim) + ".sdf") as w:
            for mol in molecules:
                w.write(mol)

        #Visualize
        if returnSVG:
            rd_kit = [mol for mol in molecules]
            SVG = Chem.Draw.MolsToGridImage(rd_kit, useSVG=True, molsPerRow=10)
            with open(output_dir + "visualization/" + cluster_name + ".svg", "w") as f:
                f.write(SVG)
    
if __name__ == '__main__':
    randomize_cluster_based(
        cluster_map = '/localhome/lconconi/CREEDS/creeds/output/FFS_cluster04_c/clustersFFS_cluster04_c_MCMS.json',
        sdf_files = '/localhome/lconconi/CREEDS/creeds/output/FFS_cluster04_c/sdf_files/',
        output_dir = '/localhome/lconconi/CREEDS/creeds/output/FFS_cluster04_c/simulation_files/')
    randomize(
        output_dir = '/localhome/lconconi/CREEDS/creeds/output/FFS_cluster04_nc/simulation_files/',
        sdf_files = '/localhome/lconconi/CREEDS/input/FreeSolv/',
        ligand_list = None,
        cluster_map_file = '/localhome/lconconi/CREEDS/creeds/output/FFS/clustersFFS_MCMS.json',
        cluster_name = "Cluster_4",
        max_num_Simulations = 10
    )
    #randomizeCluster('..//', 'clusters.json', 'randomized/', )