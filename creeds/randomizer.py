from itertools import combinations
from random import sample
from typing import List, Optional
import json
import numpy as np
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

def find(i, parent):
    '''
    Find the set of vertex i
    '''
    while parent[i] != i:
        i = parent[i]
    return i

def union(i, j, parent):
    '''
    Does the union of two sets. It returns false if i and j are already in the same set
    '''
    a = find(i, parent)
    b = find(j, parent)
    parent[a] = b



def randomize_cluster_based(cluster_map : str, 
                            sdf_files : str, 
                            output_dir: str, 
                            simulation_cluster_map = "simClusterMap.json",
                            max_num_simulations: int = 10,
                            fix_simulation_num : bool = True,
                            fix_ligand_num : bool = True,
                            numLigands_per_sim : int = 15,
                            returnSVG = True,
                            simpleOverlap: bool = False,
                            distanceMatrix : str = "FFS_cluster04.npy",
                            ID_file : str = "FFS_cluster04_IDs.json"
                            ):
    '''
    takes in a cluster_map, returns a partition with overlap to simulate in REEDS

    parameters:
        cluster_map: json file with the cluster map
        sdf_files: directory with the sdf files
        output_dir: directory to save the simulation files
        simulation_cluster_map: json file to save the simulation cluster map
        max_num_simulations: maximum number of simulations to perform
        fix_simulation_num: if True, the number of simulations will be fixed to max_num_simulations
        fix_ligand_num: if True, the number of ligands per simulation will be fixed to numLigands_per_sim
        numLigands_per_sim: number of ligands per simulation
        returnSVG: if True, return SVG files of the aligned molecules
        simpleOverlap: if True, the ligands will be overlapping by a simple chain
        distanceMatrix: distance matrix of the ligands used in the clustering phase
        ID_file: json file with the IDs of the ligands used in the clustering phase
        
    '''
    # create random overlapping sets of numLigands
    # num Ligands is meant without the overlap
    if os.path.exists(cluster_map):
        cluster_json = json.load(open(cluster_map))
    else:
        print("Cluster map does not exist")
        return -1
    
    # check how many clusters were generated
    # if there are too many Ligands in one cluster depending on the mode split them randomly or throw an exception to the user
    # if there are too many Clusters based on the mode clump the fewest ones together or throw an exception to the user

    if fix_simulation_num:
        if len(cluster_json.keys()) <= max_num_simulations:
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
                print("Cluster: ", cluster_name)
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
    if simpleOverlap: # Simple
        simulation_keys = list(simulation_ligands.keys())
        for i, cluster in enumerate(simulation_keys):
            next_key = (i + 1) % num_simulations
            next_cluster_name = simulation_keys[next_key]
            simulation_ligands[cluster].append(simulation_ligands[next_cluster_name][0])
    else: # Nearest neighbour MST

        print("loading distance Matrix")
        dists = np.load(distanceMatrix)
        
        print("loading IDs")
        IDs = json.load(open(ID_file))

        simulation_keys = list(simulation_ligands.keys())
        # cluster distances
        cluster_dists = np.ones((len(simulation_keys), len(simulation_keys)))
        
        coresponding_ligands = {}

        # iterate over all linkages
        for i, cluster_a in enumerate(simulation_keys):
            for j, cluster_b in enumerate(simulation_keys):
                # skip if the same cluster
                if i == j:
                    continue

                # minimal distance outside of cluster
                min = None

                min_lig_a = None
                min_lig_b = None

                # iterate over all ligands in the two clusters
                for ligand_a in simulation_ligands[cluster_a]:
                    for ligand_b in simulation_ligands[cluster_b]:
                        # get the distance between the two ligands
                        dist = dists[IDs[ligand_a], IDs[ligand_b]]
                        if min is None or dist < min:
                            min = dist
                            min_lig_a = ligand_a
                            min_lig_b = ligand_b

                cluster_dists[i][j] = min
                coresponding_ligands[(cluster_a, cluster_b)] = (min_lig_a, min_lig_b)

        # find mst of the cluster_dists
        # Kruskal's algorithm

        cost = 0
        parent = np.zeros(len(simulation_keys), dtype=int)
        
        # initialize the parent array
        for i in range(len(simulation_keys)):
            parent[i] = i
        
        edge_count = 0

        while edge_count < len(simulation_keys) - 1:
            min = None
            a = -1
            b = -1
            for i in range(len(simulation_keys)):
                for j in range(len(simulation_keys)):
                    if find(i, parent) != find(j, parent) and (min is None or cluster_dists[i][j] < min):
                        min = cluster_dists[i][j]
                        a = i
                        b = j
            
            #TODO: add the ligands to the simulation_ligands

            union(a, b, parent)
            edge_count += 1
            cost += min
            if len(simulation_ligands[simulation_keys[a]]) > len(simulation_ligands[simulation_keys[b]]):
                simulation_ligands[simulation_keys[b]].append(coresponding_ligands[(simulation_keys[a], simulation_keys[b])][0])
            else:
                simulation_ligands[simulation_keys[a]].append(coresponding_ligands[(simulation_keys[a], simulation_keys[b])][1])

        print("additional cost ", cost)
        print("parent array ", parent)
        print("edge_count ", edge_count)
        print("coresponding ligands ", coresponding_ligands)
                

    #write simulation cluster map to json and save it to file
    sim_map_path = os.path.join(output_dir, simulation_cluster_map)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(sim_map_path, "w") as json_out:
        json_sim_map = json.dumps(simulation_ligands)
        json_out.write(json_sim_map)

    #concatenate single sdf files into the specific clustered sdf files

    for cluster_name in simulation_ligands.keys():
        cluster_path = os.path.join(output_dir, cluster_name + ".sdf")
        with open(cluster_path, "w") as sim_sdf:
            for ligand in simulation_ligands[cluster_name]:
                lig_path = os.path.join(sdf_files, ligand + ".sdf")
                lig = open(lig_path, "r")
                sim_sdf.write(lig.read())
    

    # align each simulation file 
    for cluster_name in simulation_ligands.keys():
        molecules = []

        data_path = os.path.join(output_dir, cluster_name + ".sdf")
        data = Chem.SDMolSupplier(data_path, removeHs=False)

        #first atom i s reference
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
            svg_folder_path = os.path.join(output_dir, "visualization")
            
            if not os.path.exists(svg_folder_path):
                os.makedirs(svg_folder_path)

            svg_path = os.path.join(svg_folder_path, cluster_name + ".svg")

            with open(svg_path, "w") as f:
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
        raise ValueError("Please provide either max_num_Ligands or num_Simulations. Furthermore check that max_num_Ligands and or num_Simulation is positive and greater one.")

    # no ligand_list and no cluster_map_file is given
    if ligand_list is None and cluster_map_file is None:
        raise ValueError("Either enter a list of ligands to be prepared to the REEDS simulation or enter the path to \
              a cluster_map with the specific cluster that should be prepared.")

    
    # no list of ligands is given but a cluster_map without a specific cluster_name 
    if (ligand_list is None) and (cluster_map_file is not None) and (cluster_name is None):
        raise ValueError("Please enter a specific cluster_name to be prepared for the REEDS simulation, \
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

    # Cluster MST
    # randomize_cluster_based(
    #     cluster_map = '/localhome/lconconi/CREEDS/creeds/output/FFS_cluster04_c_mst_noise/cluster04_c_mst_noise.json',        
    #     sdf_files = '/localhome/lconconi/CREEDS/creeds/output/FFS_cluster04_c/sdf_files/',
    #     output_dir = '/localhome/lconconi/CREEDS/creeds/output/FFS_cluster04_c_mst_noise/',
    #     simpleOverlap = False,
    #     distanceMatrix = '/localhome/lconconi/CREEDS/creeds/output/FFS_cluster04_c_mst_noise/FFS_cluster04.npy',
    #     ID_file= '/localhome/lconconi/CREEDS/creeds/output/FFS_cluster04_c_mst_noise/cluster04_c_mst_noise_ids.json',
    #     fix_ligand_num = False,
    #     fix_simulation_num = False,
    #     numLigands_per_sim= 20
    #     )
    
    # Cluster p38 spectral
    # randomize_cluster_based(
    #     cluster_map='/localhome/lconconi/CREEDS/creeds/output/p38/p38_clusters_spectral.json',
    #     sdf_files = '/localhome/lconconi/CREEDS/input/p38/',
    #     output_dir = '/localhome/lconconi/CREEDS/creeds/output/p38/spectral',
    #     simpleOverlap = True,
    #     distanceMatrix = '/localhome/lconconi/CREEDS/creeds/output/p38/distance_matrix.npy',
    #     ID_file= '/localhome/lconconi/CREEDS/creeds/output/p38/p38_ids.json',
    #     fix_ligand_num=False,
    #     fix_simulation_num=False)

    # #Cluster p38 dbscan
    # randomize_cluster_based(
    #     cluster_map='/localhome/lconconi/CREEDS/creeds/output/p38/p38_clusters_dbscan.json',
    #     sdf_files = '/localhome/lconconi/CREEDS/input/p38/',
    #     output_dir = '/localhome/lconconi/CREEDS/creeds/output/p38/dbscan',
    #     simpleOverlap = True,
    #     distanceMatrix = '/localhome/lconconi/CREEDS/creeds/output/p38/distance_matrix.npy',
    #     ID_file= '/localhome/lconconi/CREEDS/creeds/output/p38/p38_ids.json',
    #     fix_ligand_num=False,
    #     fix_simulation_num=False)

    # Unclustered 
    # randomize(
    #     output_dir = '/localhome/lconconi/CREEDS/creeds/output/p38/unclustered/',
    #     sdf_files = '/localhome/lconconi/CREEDS/input/p38/',
    #     cluster_map_file = '/localhome/lconconi/CREEDS/creeds/output/p38/unclustered/ligand_list.json',
    #     cluster_name = "Cluster_0",
    #     max_num_Ligands = 30,
    #     max_num_Simulations= 4,
    #     returnSVG = True
    # )

    # HIF2A

    # print("Create SDF Files for clustered dbscan simulations")
    # randomize_cluster_based(
    #     cluster_map = '/localhome/lconconi/CREEDS/creeds/output/hif2a/hif2a_clusters_dbscan.json',
    #     sdf_files = '/localhome/lconconi/CREEDS/input/hif2a/',
    #     output_dir = '/localhome/lconconi/CREEDS/creeds/output/hif2a/dbscan',
    #     simpleOverlap = True,
    #     distanceMatrix = '/localhome/lconconi/CREEDS/creeds/output/hif2a/distance_matrix.npy',
    #     ID_file = '/localhome/lconconi/CREEDS/creeds/output/hif2a/hif2a_ids.json',
    #     fix_ligand_num = False,
    #     fix_simulation_num = False
    # )

    # print("Create SDF Files for clustered spectral simulations")
    # randomize_cluster_based(
    #     cluster_map = '/localhome/lconconi/CREEDS/creeds/output/hif2a/hif2a_clusters_spectral.json',
    #     sdf_files = '/localhome/lconconi/CREEDS/input/hif2a/',
    #     output_dir = '/localhome/lconconi/CREEDS/creeds/output/hif2a/spectral',
    #     simpleOverlap = True,
    #     distanceMatrix = '/localhome/lconconi/CREEDS/creeds/output/hif2a/distance_matrix.npy',
    #     ID_file = '/localhome/lconconi/CREEDS/creeds/output/hif2a/hif2a_ids.json',
    #     fix_ligand_num = False,
    #     fix_simulation_num = False
    # )

    # print("create sdf files for unclustered simulations")
    # randomize(
    #     output_dir = '/localhome/lconconi/CREEDS/creeds/output/hif2a/unclustered/',
    #     sdf_files = '/localhome/lconconi/CREEDS/input/hif2a/',
    #     cluster_map_file = '/localhome/lconconi/CREEDS/creeds/output/p38/unclustered/ligand_list.json',
    #     cluster_name = "Cluster_0",
    #     max_num_Ligands = 30,
    #     max_num_Simulations= 4,
    #     returnSVG = True
    # )

    # DOMEN
    concatenate(
        ClusterMap = '/localhome/lconconi/CREEDS/creeds/output/FFS/clustersFFS_MCMS.json',
        clusterSDF_dir = '/localhome/lconconi/CREEDS/input/FreeSolv/',
        output_dir = '/localhome/lconconi/CREEDS/creeds/output/FFS/'
    )