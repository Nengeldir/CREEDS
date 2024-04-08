from random import sample
from typing import List
import json

def randomize_cluster_based(cluster_file : str, output_dir: str, randomizeCluster : str = "Cluster_4.sdf", numLigands : int = 15):
    '''
    takes in a set of ligands and returns a partition with overlap to simulate in REEDS

    parameters
        cluster_file
        output_dir
        randomize_Cluster
    '''
    # create random overlapping sets of numLigands
    # num Ligands is meant without the overlap 
    f = open(cluster_file, 'r')
    cluster_data = json.load(f)

    simulationLigands = {}
    numSimulations = 0
    if len(cluster_data[randomizeCluster]) % numLigands == 0:
        numSimulations = len(cluster_data[randomizeCluster]) // numLigands
    else:
        numSimulations = len(cluster_data[randomizeCluster]) // numLigands + 1

    # clusters to randomize
    for i in range(numSimulations):
        simulationLigands[i] = sample(cluster_data[cluster], numLigands)
        cluster_data[cluster].remove(simulationLigands[i])

    # add the overlap to all existing clusters
    # choose the first ligand in the next cluster to be the overlap
    for i, key in enumerate(simulationLigands.keys):
        next_key = (i + 1) % numSimulations
        simulationLigands[key].append(simulationLigands[next_key][0])
        
    print(simulationLigands)
    print(cluster_data)

    #clusters to concatenate
    
    for cluster in simulationLigands.keys:
        with open(output_dir + "/Simulation" + str(cluster) + '.sdf', 'w') as f:
            for item in cluster_data[cluster]:
                print(item, end=", ")
                with open(directory + "/" + item + ".sdf", 'r') as file:
                    f.write(file.read())

if __name__ == '__main__':
    randomizeCluster('..//', 'clusters.json', 'randomized/', )