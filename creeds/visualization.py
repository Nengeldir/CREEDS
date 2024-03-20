import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def drawClusters(sub_IDs : map):
    G = nx.Graph()
    
    # Add nodes
    for cluster in sub_IDs:
        for i, ligand in enumerate(sub_IDs[cluster]):
            G.add_node(ligand)
            if i == 0:
                first_ligand = ligand
            else: 
                G.add_edge(first_ligand, ligand)
    
    nx.draw(G)
    plt.show()

if __name__ == "__main__":
    # Test the function
    sub_IDs = {0: ['mobley_1520842', 
                   'mobley_1873346', 
                   'mobley_20524', 
                   'mobley_3398536', 
                   'mobley_4287564', 
                   'mobley_4483973', 
                   'mobley_4883284', 
                   'mobley_5977084', 
                   'mobley_7599023', 
                   'mobley_7608462'], 
                1: ['mobley_3187514', 
                    'mobley_4035953', 
                    'mobley_9478823']}
    
    drawClusters(sub_IDs)
    