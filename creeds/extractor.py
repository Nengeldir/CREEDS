import json
import os
import shutil

def extract(json_cluster : str, input_sdf_dir : str, output_sdf_dir : str, cluster_name:str):
    '''
    Give a cluster.json and a specific cluster, this function will then extract the sdf files from a folder 
    and copy it to another location for further processing

    Parameters:
        json_cluster: str
            path to the cluster.json file that contains the clustering information with the ligands
        input_sdf_dir: str
            path to the sdf directory that were used in the creation of the clusters
        output_sdf_dir: str
            path to the sdf directory where it should be copied
        cluster: str
            name of the cluster, which should be extracted
    '''
    json_file = open(json_cluster)
    clusters = json.load(json_file)

    for ligand in clusters[cluster_name]:
        file_path_input = os.path.join(input_sdf_dir, ligand + ".sdf")
        file_path_output = os.path.join(output_sdf_dir, ligand + ".sdf")
        shutil.copyfile(file_path_input, file_path_output)

if __name__ == '__main__':
    extract('/localhome/lconconi/CREEDS/creeds/output/FFS/clustersFFS_MCMS.json', '/localhome/lconconi/CREEDS/input/FreeSolv', 
            '/localhome/lconconi/CREEDS/creeds/output/FFS_cluster04_nc/sdf_files', 'Cluster_4')
    