from typing import List
import json

def concatenate(directory: str, cluster_file : str, output_dir: str, includeList : List[int] = None):
        
    f = open(cluster_file, 'r')
    cluster_data = json.load(f)

    print(cluster_data)

    #clusters to concatenate
    if includeList is None:
        includeList = list(cluster_data.keys())
    
    for cluster in includeList:
        print(cluster)
        
        with open(output_dir + "/" + str(cluster) + '.sdf', 'w') as f:
            for item in cluster_data[cluster]:
                print(item, end=", ")
                with open(directory + "/" + item + ".sdf", 'r') as file:
                    f.write(file.read())


if __name__ == '__main__':
    concatenate('../input/FreeSolv', 'clusters.json', 'output/')
     