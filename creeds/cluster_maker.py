import lomap
import json
import os
import numpy as np
from typing import List, Optional
from tempfile import TemporaryFile

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.backends.backend_pdf
import matplotlib.ticker as tick

from rdkit import Chem
from rdkit.Chem import rdMolAlign
from rdkit.Chem import AllChem
from rdkit.Chem import rdFMCS

from utils import _clean_NaN, _getID, \
    _clean_ID_list, _k_dist, _find_max_curvature, _dbscan, _sub_arrays, _spectral


from scipy import interpolate

import argparse

class ClusterMaker():

    def __init__(self,
                 filePath_: str,
                 loadMatrix_: bool = False,
                 loadFile_: str = "distance_matrix.npy",
                 output_file_: str = "clusters.json",
                 method_: str = "MCS",
                 plot_folder_: str = "plots/",
                 **kwargs
                 ):
        '''
        This class is used to create clusters of ligands based on their similarity scores.
        The similarity scores are calculated using the Maximum Common Substructure (MCMS) and the Shape score (shape).
        By default it uses the MCMS score calculated in the lomap package.

        The class uses an instance of lomap.DBMolecules to either handle the molecules and/or calculate the Maximum Common Substructure.
        
        The initialization takes in a vast variety of parameters. The user can easily adapt the input to cater to their needs. A detailed description of each
        parameter can be found in the bottom portion of this docstring.

        The only necessary parameter is the filePath: where the sdf files are located (Note: they should be separate for each molecule)

        The ClusterMaker class has a variety of methods. It can calculate, save and load the distance matrix, create clusters, get the DBMolecules object, and
        calculate the clusters. The class uses a helper class called Plotter to plot the distance matrix and
        the clusters.

            Parameters
                filePath: str, 
                    Description: The path to the directory containing the ligand files. It should contain single sdf/md2 files.
                    Default: no-default
                parallel_: int
                    Description: The number of parallel processes to use. It generally speeds up the calculation.
                    Default = 4
                verbose_: str, 
                    Description: The verbosity level. Modes one of: 'off', 'info', 'pedantic'
                    Default = 'off'
                time_: int, 
                    Description: The time limit for the calculation.
                    Default: 20
                ecrscore_: float
                    Description: The electrostatic score when two molecules have different charges (it is used if ecrscore_ != 0)
                    Default: 0.0
                threed_: bool
                    Description: If true, symmetry-equivalent MCSes are filtered to prefer the one with the best real-space alignment
                    Default: False
                max3d_: float
                    Description: The MCS is filtered to remove atoms which are further apart than this threshold.
                    Default: 1000.0
                element_change_: bool
                    Description: Whether to allow changes in elements between two mapping.
                    Default: True
                output_: bool
                    Description: whether to output the results.
                    Default: False
                name_: str
                    Description: the prefix name of the output file
                    Default: CREEDS_out
                loadMatrix_: bool
                    Description: Flag to determine if the distance matrix should be loaded instead of calculating it by scratch
                    Default: False
                loadFile_: str
                    Description: Path to a npy (numpy binary file), that contains the distance matrix. Loading a precalculated distance matrix
                        prevents recalculating it again and therefore saves time.
                    Default: distance_matrix.npy
                output_file_: str
                    Description: The filename and path where the clusters should be saved, by default it generates a json file where each key is the cluster name
                        Followed by a list of ligands. 
                    Default: clusters.json
                method_: str
                    Description: Which method should be used to calculate the distance matrix. Choose one of MCSS (Maximum Common Substructure Score) / Shape (RDKit
                    Alignment Score)
                    Default: MCSS
                plot_folder_: str
                    Description: Path to the folder where all plots are saved.
                    Default: plots/

                LoMap Specific Options and Variables
                    output_no_images_: bool
                        Description: A flag to disable generation of output images.
                        Default: False
                    output_no_graph_: bool
                        Description: A flag to disable generation of output graph (.dot file).
                        Default: False
                    display_: bool
                        Description: A flag to determine wheter to display or not a network made by matplotlib.
                        Default: False
                    allow_tree_: bool
                        Description: if set, then the final graph does not need a cycle covering and therefore will be a tree.
                        Default: False
                    max_: int
                        Description: The maximum diameter of the resulting graph
                        Default: 6
                    cutoff_: float
                        Description: The Minimum Similarity Score(MSS) used to build the graph.
                        Default: 0.4
                    radial_: bool
                        Description: Option to use the radial graph approach in LoMap
                        Default: False
                    hub_: str
                        Description: Manual selected compound that is the center of the graph
                        Default: None
                    fast_: bool
                        Description: Using the fast graphing option
                        Default: False
                    links_file_: str
                        Description: the name of a file containing links to seed the graph with.
                        Default: None
                    known_actives_file_: str
                        Description: the name of a file containing mols whose actviity is known.
                        Default: None
                    max_dist_from_actives_: int
                        Description: The maximum number of links from any molecule to an active
                        Default: 2
                    use_common_core_: bool
                        Description: Wheter to search among all input molecules for a common core to speed up pairwise MCS calculations.
                        Default: True

            Returns
                Instance of ClusterMaker object.
        '''

        

        self.db_mol = lomap.DBMolecules(directory = filePath_, 
                                        parallel = kwargs.get('parallel', 4),
                                        verbose = kwargs.get('verbose', 'off'),
                                        time = kwargs.get('time', 20),
                                        ecrscore = kwargs.get('ecrscore', 0.0),
                                        threed = kwargs.get('threed', False),
                                        max3d = kwargs.get('max3d', 1000.0),
                                        element_change = kwargs.get('element_change', True),
                                        output = kwargs.get('output', False),
                                        name = kwargs.get('name', "CREEDS_out"),
                                        output_no_images = kwargs.get('output_no_images', False),
                                        output_no_graph = kwargs.get('output_no_graph', False),
                                        display = kwargs.get('display', False),
                                        allow_tree = kwargs.get('allow_tree', False),
                                        max = kwargs.get('max', 6),
                                        cutoff = kwargs.get('cutoff', 0.4),
                                        radial = kwargs.get('radial', False),
                                        hub = kwargs.get('hub', None),
                                        fast = kwargs.get('fast', False),
                                        links_file = kwargs.get('links_file', None),
                                        known_actives_file = kwargs.get('known_actives_file', None),
                                        max_dist_from_actives = kwargs.get('max_dist_from_actives', 2),
                                        use_common_core = kwargs.get('use_common_core', True),
                                        shift = kwargs.get('shift', True)
                                        )
        
        
        if loadMatrix_:
            if os.path.exists(loadFile_):
                print("Loading Similarity Matrix from file: ", loadFile_)
                self.sim_data_ = np.load(loadFile_)
            else:
                raise ValueError("Similarity File does not exist at given location ", loadFile_)
        else:
            if method_ == "MCSS":

                print("Calculating Similarity Matrix by using Maximum Common Substructure...\n")
                
                self.strict_, _ = self.db_mol.build_matrices()
                self.sim_data_ = self.strict_.to_numpy_2D_array()
                
                print("Finished calculation of Similarity Matrix.")

            elif method_ == "Shape":

                print("Calculating Similarity Matrix by using the shape method... \n")

                self.sim_data_ = self.calculateShapeMatrix(self.db_mol)

                print("Finished calculation of Similarity Matrix.")

            elif method_ == "MCES":

                print("Cacluating Similarity Matrix by using MCES... \n")
                
                self.calculateMCES(self.db_mol)

                print("Finished calculation of Similarity Matrix.")

            else:
                raise ValueError("Invalid method. Please use 'MCSS', 'Shape', 'MCES'")
        
        self.filePath_ = filePath_
        self.output_file_ = output_file_
        self.loadFile_ = loadFile_
        self.n_arr = _clean_NaN(self.sim_data_)
        self.ID_List_ = _clean_ID_list(_getID(self.db_mol, self.n_arr))
        self.plot_folder_ = plot_folder_
        self.plotter_ = Plotter(self.ID_List_, self.plot_folder_)
        self.sub_arrs_ = None
        self.sub_IDs_ = None
        self.clustering_ = None
    

    def calculateShapeMatrix(self, db_mol):
        '''
        This function is used to calculate the shape matrix for the ligands.
        The function uses RDKit to calculate the shape matrix for the ligands.
        The function returns the shape matrix for the ligands.

            Parameters:
                self: object, the ClusterMaker object.
                db_mol: object, the lomap.DBMolecules object.

            Returns:
                shape matrix for the ligands.
        '''
        #TODO: test
        self.sim_data_ = lomap.SMatrix(shape=(len(db_mol._list), ))
        for i, mol_a in enumerate(db_mol._list):
            for j, mol_b in enumerate(db_mol._list):
                if i == j:
                    self.sim_data_[i, j] = 1
                else:
                    mol_i = mol_a.getMolecule()
                    mol_j = mol_b.getMolecule()

                    Chem.SanitizeMol(mol_i)
                    Chem.SanitizeMol(mol_j)

                    pyO3A = rdMolAlign.GetO3A(prbMol = mol_i, refMol = mol_j)
                    score = pyO3A.Align()
                    
                    self.sim_data_[i, j] = score
                    self.sim_data_[j, i] = score
                    print(score)
        

        return self.sim_data_

    def calculateMCES(self, db_mol):
        '''
        This function is used to calculate the MCES score for the ligands.
        The function uses the lomap package to calculate the MCES score for the ligands.
        The function returns the MCES score for the ligands.

            Parameters:
                self: object, the ClusterMaker object.
                db_mol: object, the lomap.DBMolecules object.

            Returns:
                MCES score for the ligands.
        '''
        # TODO:
        print(db_mol._list)
        pass
    
    def saveDistanceMatrix(self, path : str) -> None:
        '''
        Save the Similartiy/Distance matrix to a file.

        Parameters:
            path (str): The path to the file where the distance matrix will be saved.

        Returns:
            None
        '''
        np.save(path, self.sim_data_)

    def get_similarity_matrix(self):
        '''
        Returns the Similarity/Distance Matrix. 
        '''
        return self.sim_data_
    
    def cluster_spectral(self, num_clusters, auto : bool = False, verbose : bool = True) -> tuple[dict, dict]:
        '''
        This function creates clusters using the K-Means algorithm. The function uses the KMeans algorithm implemented in
        the sklearn package.

        Parameters:
            self: object, the ClusterMaker object.
            auto: bool, if true the auto cutoff is taken
            verbose: bool, if true more verbosity is outputed

        Returns:
            sub_arrs: dict, n_arr subdivided into dict of cluster number keys
            sub_IDs: dict, ID_list subdivided into dict of cluster number keys
        '''
        self.data = 1 - self.sim_data_
        labels, mask = _spectral(self.data, verbose = verbose, num_clusters = num_clusters)

        ax = self.plotter_.plt_cluster(self.data, labels)
        
        # Output table for user cluster selection
        if verbose:
            plt.show()
    
        n_clusters_ = len(labels)
        # Figure saving
        output_path = os.path.join(self.plot_folder_, "output.pdf")
        pdf = matplotlib.backends.backend_pdf.PdfPages(output_path)
        fig1 = self.plotter_.plt_cluster(self.data, labels)
        fig2 = self.plotter_.plt_dbscan(self.data, labels, mask, n_clusters_)
        pdf.savefig(fig1, bbox_inches= "tight", pad_inches=0.5)
        pdf.savefig(fig2)
        pdf.close()

        # Generate sub-arrays of clusters. Stored in dictionaries.
        sub_arr, sub_ID = _sub_arrays(labels, self.sim_data_, self.ID_List_)

        return sub_arr, sub_ID

    
    def create_clusters(self, interactive: bool = True, algorithm : str = "dbscan", verbose : bool = True, num_clusters : int = -1) -> json:
        '''
        This function is used to create clusters of ligands based on their similarity scores. There needs to be a sensible distancematrix saved in self.simdata_
        It uses the Plotter class to plot the distance matrix and the clusters.
        The function saves the clusters to a json file. After the function call following variables are now populated
            self.sub_arrs_: dict, n_arr subdivided into dict of cluster number keys
            self.sub_IDs_: dict, ID_list subdivided into dict of cluster number key
            self.clustering_: json, parsed json object that contains the clustering information 

        Parameters:
            self: object, the ClusterMaker object.
            interactive: bool, if true the auto cutoff/cluster number is taken
            algorithm: str, the algorithm used for clustering, either dbscan or K-Means
        Returns:
            Clustering: json, parsed json object that contains the clustering information, it is also saved in the class
            for later use.
        '''

        # Start interactive sequence to determine clusters, clusters are saved in sub_IDs, where sub_arrs gets the distance matrix of the cluster
        
        if algorithm not in ["dbscan", "spectral"]:
            raise ValueError("Invalid algorithm. Please use 'dbscan' or 'K-Means'")
        
        if algorithm == "dbscan":
            self.sub_arrs_, self.sub_IDs_ = self.cluster_interactive(auto=interactive, verbose = verbose)
        elif algorithm == "spectral":
            self.sub_arrs_, self.sub_IDs_ = self.cluster_spectral(auto = interactive, verbose = verbose, num_clusters = num_clusters)

        clustering = "{"

        for key in self.sub_IDs_.keys():
            clustering += "\"Cluster_" + str(key) + "\"" + " : [" 

            for cluster in self.sub_IDs_[key]:
                clustering += "\"" + str(cluster) + "\""
                if cluster != list(self.sub_IDs_[key])[-1]:
                    clustering += ", "
            clustering += "]"
            if key != len(self.sub_IDs_.keys()) - 1:
                clustering += ",\n"


        clustering += "}"

        self.clustering_ = json.loads(clustering)

        with open(self.output_file_, 'w') as f:
            f.write(clustering)

        print("Wrote Clustering to ", self.output_file_, " as json.")

        return self.clustering_
        
    def getDBMolecules(self):
        '''
        Returns the lomap.DBMolecules object. Useful if one wants to access lomap functionalities
            
            Parameters:
                self: object, the ClusterMaker object.
            
            Returns:
                db_mol: object, the lomap.DBMolecules object.
        '''
        return self.db_mol
    
    def cluster_interactive(self, verbose: bool = True, auto : bool = False) -> tuple[dict, dict]:
        '''
        Function to inspect distance data and self assign the clustering neighbour distance cutoff. This is the current default.
        # TODO: Implement automatic version and version where one can specify number of clusters

            Parameters:
                self: object, the ClusterMaker object.
                verbose: bool, if true more verbosity is outputed
                auto: bool, if true the auto cutoff is taken

            Returns:
                sub_arrs: dict, n_arr subdivided into dict of cluster number keys
                sub_IDs: dict, ID_list subdivided into dict of cluster number keys
            
        '''
        def get_numeric(splitme):
            '''
            Helper function that reads string and returns integer values.
            '''
            delimiters = [" ", ",", "[", "]", ")", "("]
            cleaned = splitme
            for i in delimiters:
                cleaned = cleaned.replace(i, " ")
            digs = [int(s) for s in cleaned.split() if s.isdigit()]
            return digs

        # Generate distance data 
        # if sim_data[i][j] = 1 we have a perfect score, or equivalently self.data = 0
        self.data = 1 - self.sim_data_

        # Output distance info to user.
        x, dists = _k_dist(self.data)
        auto_cutoff = _find_max_curvature(x, dists, plot_path=self.plot_folder_, savefigs=True, verbose=verbose)

        print("auto_cutoff was determined to be ", auto_cutoff)

        if auto:
            print("Taking auto cutoff as auto is set to True.")
            user_cutoff = auto_cutoff
        else:
            # Get user input on distance cutoff, epsilon in dbscan, typed
            input_cutoff = input("Enter a neighbor distance cutoff for clustering:")
            if input_cutoff == "":
                # If enter, set to autodetected cutoff
                user_cutoff = auto_cutoff
            else:
                user_cutoff = float(input_cutoff)
        
        # Do Clustering according to dbscan (Note that dbscan normally sets noise to cluster -1)
        labels, mask, n_clusters_ = _dbscan(self.data, dist_cutoff=user_cutoff, min_s = 2)
        
        ax = self.plotter_.plt_cluster(self.data, labels)
        
        # Output table for user cluster selection
        if verbose:
            plt.show()
    

        # Figure saving
        output_path = os.path.join(self.plot_folder_, "output.pdf")
        pdf = matplotlib.backends.backend_pdf.PdfPages(output_path)
        fig1 = self.plotter_.plt_cluster(self.data, labels)
        fig2 = self.plotter_.plt_dbscan(self.data, labels, mask, n_clusters_)
        pdf.savefig(fig1, bbox_inches= "tight", pad_inches=0.5)
        pdf.savefig(fig2)
        pdf.close()

        # Generate sub-arrays of clusters. Stored in dictionaries.
        sub_arr, sub_ID = _sub_arrays(labels, self.sim_data_, self.ID_List_)

        return sub_arr, sub_ID

    def clusters_w_ref(ref_ligs, sub_ID):
        '''
        # TODO:
        Find which clusters contain reference ligands.
        Input a list of reference ligand names and check
        which cluster it is found in in dict sub_ID.

        Outputs: list of clusters containing a ref lig.

            Parameters:
                ref_ligs: list of reference ligands.
                sub_ID: dictionary, ligand IDs subdivided into clusters.

            Returns:
                cluster_set: list of clusters containing a ref lig.
                sub_refs: dictionary of reference ligands places by key
                          where key is the cluster number.
        '''
        hits = []
        # Make dictionary with cluster number keys to store ref ligs
        keyList = set(sub_ID.keys())
        sub_refs = {key: [] for key in keyList}
        # Find which clusters contain reference ligands.
        for i in ref_ligs:
            for k in sub_ID:
                # If ref lig name in list
                if i in sub_ID[k]:
                    hits.append(k)
                    # Return unique cluster keys.
                    cluster_set = list(set(hits))
                    # If ref in cluster, append to dict
                    sub_refs[k].append(i)
        for k in keyList:
            if not sub_refs[k]:
                # Delete keys w empty lists.
                del sub_refs[k]
        return cluster_set, sub_refs
    
    def writeIdList(self, ID_File):
        '''
        This function writes out the ID_Files as a json file. It is useful for the randomizer module which needs the IDs of the ligands.
        The structure is the following: 
        {
            ligand_1 : 1
            ligand_2 : 2
            .
            .
            .
            ligand_N : N
        }
        '''

        output = "{"

        for i, ligand in enumerate(self.ID_List_):
            output += "\"" + ligand + "\"" + " : " + str(i)

            if i != len(self.ID_List_) - 1:
                output += ",\n"


        output += "}"

        json_object = json.loads(output)
        #self.clustering_ = json.loads(output)

        with open(ID_File, 'w') as f:
            f.write(output)


class Plotter():
    '''
    This class is used to plot the distance matrix and the clusters.
    The class has methods to plot the heatmap of the distance matrix, the clusters, and the cluster regions.
    The class is used by the ClusterMaker class to plot the
    distance matrix and the clusters.
    '''
    

    def __init__(self, id_list, plot_folder = "plots"):
        self.ID_list = id_list
        self.plot_folder_ = plot_folder
    def plt_heatmap(self, data, ax, fig, **kwargs):
            '''
            Plots a heatmap of the chemical distances that will be clustered.
            
                Parameters:
                    data: symmetric distance array (1 - similarity)
                    
                Optional Parameters:
                    cmap: color to plot with, default is matplotlib 'CMRmap'
                    tick_interval: default produces 15 or fewer ticks.
                    
                Returns:
                    ax: the subfigure.
            '''
            
            # Ensure input data has 0 distance on diagonal
            np.fill_diagonal(data, 0.0)
            # Get number of ligands
            N = data.shape[0]
            # Define optional arugments
            # Plot coloring
            cmap = kwargs.get('cmap', 'inferno')

            # X and Y ticks
            # Temp note: I just deleted , 10 as the default in get kwargs
            #tick_interval = kwargs.get('tick_interval')
            #if tick_interval is None:
                # The default is to have 15 or fewer ticks.
            #if N > 15:
            tick_interval = N//15 + 1
            #else:
                #tick_interval = 1
            #else:
                #tick_interval = tick_interval

            # Heatmap plot
            im = ax.imshow(data, origin='upper', cmap='CMRmap')
            
            # Add residue ID labels to axes.
            ax.set_yticks(np.arange(N)[::tick_interval])
            ax.set_xticks(np.arange(N)[::tick_interval])
            ax.set_yticklabels(self.ID_list[::tick_interval])
            ax.set_xticklabels(self.ID_list[::tick_interval], rotation = 45, ha='right', rotation_mode='anchor')

            # Add figure labels and titles.
            plt.ylabel('Ligand IDs')
            plt.xlabel('Ligand IDs')
            plt.title('Distances in Similarity', pad=20)

            # Colorbar controls.
            im.set_clim(0, 1)
            cbar = fig.colorbar(im)
            cbar.ax.set_ylabel('Distance')
            return ax


    def plt_dbscan(self, data, labels, core_samples_mask, n_clusters_):
        '''
        Plots clusters on 2d plot to see distance between clusters.
        
            Parameters:
                data: symmetric distance array (1 - similarity)
                labels: array of cluster numbers by ligand
                core_samples_mask: filters clusters
                n_clusters_: the number of clusters
                
            Returns:
                fig: the figure.
        '''
        fig, ax3 = plt.subplots()
        fig.set_size_inches(6, 6)
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]
            class_member_mask = labels == k
            # Plot clusters.
            xy = data[class_member_mask & core_samples_mask]
            plt.plot(
                xy[:, 0],
                xy[:, 1],
                "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=14,
            )
            # Plot noise points.
            xy = data[class_member_mask & ~core_samples_mask]
            plt.plot(
                xy[:, 0],
                xy[:, 1],
                "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=6,
            )
        plt.title("Estimated number of clusters: %d" % n_clusters_)
        dbscan_path = os.path.join(self.plot_folder_, "dbscan_plot.png")
        plt.savefig(dbscan_path, dpi=300)
        return fig


    def plt_cluster_regions(self, labels, fig,**kwargs):
        '''
        Plots the cluster regions relative to heatmap of the chemical distances.
        
            Parameters:
                labels: calculated by DBSCAN
                
            Optional Parameters:
                cmap: color to plot with, default is matplotlib 'CMRmap'
                tick_interval: default produces 15 or fewer ticks.
                
            Returns:
                ax: the subfigure.
        '''

        N = len(self.ID_list)
        print(f'The number of ligands is {N}')
        # Define optional arugments
        # Plot coloring
        cmap = kwargs.get('cmap', 'inferno')

        # Ticks parameters
        #tick_interval = kwargs.get('tick_interval')
        #if tick_interval is None:
            # The default is to have 15 or fewer ticks.
        #if N > 15:
        tick_interval = N//15 + 1
        #else:
            #tick_interval = 1
        #else:
            #tick_interval = tick_interval
        # Option to pass ax
        ax = kwargs.get('ax')
        
        if ax is None:
            fig, ax = plt.subplots()
            fig.set_size_inches(4, 6)
        else:
        
            ax = ax
            fig = fig
            
        # Start plotting.
        # Black removed and is used for noise instead.
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        # Make the labels a 2D array that can be plotted
        labels_arr = labels.reshape(N, 1)
        # Define discrete colors for N clusters.
        cmap = plt.get_cmap('inferno')
        # Plot. Had 0.04 for the aspect before
        psm = ax.imshow(labels_arr, cmap=cmap, rasterized=True, aspect= 'auto')
        
        # Control color bar.
        cbar = fig.colorbar(psm, ax=ax, ticks=sorted(list(unique_labels)))
        
        # What I want is to just replace -1 with noise:
        '''
        labels_l = sorted(list(unique_labels))
        for i in range(len(labels_l)):
            if labels_l[i] == -1:
                labels_l[i] = 'noise'

        if -1 in unique_labels:
            #strings = [str(x) for x in sorted(list(unique_labels))[1:]]
            cbar.ax.set_yticklabels(['noise', '0', '1', '2'])
        '''
        cbar.set_label('Cluster Number', rotation=270, labelpad = 15)
        # Add residue ID labels to axes
        ax.set_yticks(np.arange(N)[::tick_interval])
        ax.set_yticklabels(self.ID_list[::tick_interval])

        ax.tick_params(
            axis='x',
            which='both',
            bottom=False,
            top=False,
            labelbottom=False)

        # Add figure labels and titles
        ax.set_ylabel('Ligand IDs')
        plt.title('Cluster Regions', pad=20)
        cluster_regions_path = os.path.join(self.plot_folder_, "cluster_regions.png")
        plt.savefig(cluster_regions_path, dpi=300, bbox_inches='tight')
        return ax
    
    # This need the kwargs options put in
    def plt_cluster(self, data, labels):
        '''
        Function to combine heatmap plot and cluster region plots into
        the same figure to compare side by side.
        
            Parameters:
                data: the distance array from similarity scores
                labels: cluster numbers calculated by DBSCAN()
                
            Returns:
                fig: the figure.
        '''
        # Clean ID list if needed, func in utils.
        _clean_ID_list(self.ID_list)
        # This was 4.5 not 5.5
        fig, axs = plt.subplots(1, 2, figsize=(9.5, 4.5), gridspec_kw={'width_ratios': [1, 4], 'height_ratios': [1]})
        plt.subplots_adjust(top=0.86, bottom=0.22)
        self.plt_cluster_regions(labels, ax=axs[0], fig=fig)
        self.plt_heatmap(data, axs[1], fig)
        axs[0].set_title('Cluster Regions', pad = 15)
        axs[1].set_title('Heatmap of chemical distance', pad = 15)
        fig.tight_layout()
        cluster_regions_heatmap_path = os.path.join(self.plot_folder_, "cluster_regions_heatmap.png")
        fig.savefig(cluster_regions_heatmap_path, dpi=300)
        return fig


   
        

if __name__ == "__main__":
    #cmaker = ClusterMaker('../input/FreeSolv')
    #cmaker.saveDistanceMatrix("distance_matrix_FullFreeSolv.npy")
    #cmaker.create_clusters()

    #matplotlib.use('TkAgg')
    #TODO: fix plots so that they are saved in the correct Folder
    cmaker = ClusterMaker('/localhome/lconconi/CREEDS/creeds/output/FFS_cluster04_c/sdf_files', 
                          loadMatrix_ = False,
                          loadFile_ = "/localhome/lconconi/CREEDS/creeds/output/FFS_cluster04_c/FFS_cluster04.npy", 
                          method_ = "MCSS", 
                          output_file_ = "/localhome/lconconi/CREEDS/creeds/output/FFS_cluster04_c_spectral/FFS_cluster04_c_MCMS_spectral.json", 
                          parallel_ = 6,
                          plot_folder_='/localhome/lconconi/CREEDS/creeds/output/FFS_cluster04_c_spectral/plots/')

    print(cmaker.ID_List_)
    cmaker.saveDistanceMatrix("/localhome/lconconi/CREEDS/creeds/output/FFS_cluster04_c_spectral/FFS_cluster04.npy")
    cmaker.writeIdList("/localhome/lconconi/CREEDS/creeds/output/FFS_cluster04_c_spectral/FFS_cluster04_IDs.json")
    cmaker.create_clusters(algorithm="spectral", num_clusters=10)

    