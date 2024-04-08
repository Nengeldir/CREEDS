import json
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
        shutil.copyfile(input_sdf_dir + "/" + ligand + ".sdf", output_sdf_dir + "/" + ligand + ".sdf")

if __name__ == '__main__':
    extract('/localhome/lconconi/CREEDS/creeds/output/FFS/clustersFFS_MCMS.json', '/localhome/lconconi/CREEDS/input/FreeSolv', 
            '/localhome/lconconi/CREEDS/creeds/output/FFS_cluster04_c/sdf_files', 'Cluster_4')
    