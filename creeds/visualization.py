import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import json

def drawClusters(clustering : map, plotFolder:str):
    '''
    Draws Subset 
    '''
    G = nx.Graph()
    pos = {}
    colors = ["blue", "green", "yellow", "orange"] * (len(clustering.keys()) // 4 + 1)
    # Add nodes
    off_number = 5
    for cluster_number, cluster_name in enumerate(clustering.keys()):
        
        if cluster_number == 0:
            offset = (0, 0)
        elif cluster_number % 4 == 0:
            offset = (off_number, off_number)
            off_number += 4
        elif cluster_number % 4 == 1:
            offset = (-off_number, off_number)
        elif cluster_number % 4 == 2:
            offset = (off_number, -off_number)
        elif cluster_number % 4 == 3:
            offset = (-off_number, -off_number)

        dis = 1
        for i, ligand_1 in enumerate(clustering[cluster_name]):
            G.add_node(ligand_1)
            if i == 0:
                pos[ligand_1] = (offset[0], offset[1])
            elif i % 4 == 0:
                pos[ligand_1] = (dis  + offset[0], offset[1])
                dis += 1
            elif i % 4 == 1:
                pos[ligand_1] = (offset[0], dis + offset[1])
            elif i % 4 == 2:
                pos[ligand_1] = (-dis + offset[0], offset[1])
            elif i % 4 == 3:
                pos[ligand_1] = (offset[0], -dis + offset[1])

            for j, ligand_2 in enumerate(clustering[cluster_name]):
                
                if i == j or i < j:
                    continue
                else: 
                    G.add_edge(ligand_1, ligand_2)

    nx.draw(G, pos, node_color = range(len(G.nodes)), cmap = plt.cm.viridis)
    plt.show()
    saveLoc = os.path.join(plotFolder, "visualization.png")
    plt.savefig(saveLoc)

if __name__ == "__main__":
    # Test the function
    clustering = json.load(open('output/FFS/clustersFFS_cluster04_c_MCMS.json', "r"))
    drawClusters(clustering, "output/FFS/plots/")
    