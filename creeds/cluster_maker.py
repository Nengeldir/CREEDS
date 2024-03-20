import lomap

import numpy as np
from typing import List, Optional
from tempfile import TemporaryFile

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.backends.backend_pdf
import matplotlib.ticker as tick

from utils import _clean_NaN, _getID, _clean_ID_list, _k_dist, _find_max_curvature, _dbscan, _sub_arrays


from scipy import interpolate

import argparse

class ClusterMaker():

    def __init__(self, 
                 filePath: str, 
                 parallel_: int = 1,
                 verbose_: str = 'off',
                 time_ : int = 20,
                 ecrscore_: float = 0.0,
                 threed_: bool = False,
                 max3d_: float = 1000.0,
                 output_: bool = False,
                 name_: str = "CREEDS_out",
                 output_no_images_: bool = False,
                 output_no_graph_: bool = False,
                 display_: bool = False,
                 allow_tree_: bool = False,
                 max_: int = 6,
                 cutoff_: float = 0.4,
                 radial_: bool = False,
                 hub_: Optional[str] = None,
                 fast_: bool = False,
                 links_file_: Optional[str] = None,
                 known_actives_file_: Optional[str] = None,
                 max_dist_from_actives_: int = 2,
                 loadMatrix: Optional[str] = False,
                 loadFile: Optional[str] = "distance_matrix.npy"
                 ):
        
        self.filePath = filePath
        self.db_mol = lomap.DBMolecules(directory = filePath, 
                                        parallel = parallel_,
                                        verbose = verbose_,
                                        time = time_,
                                        ecrscore = ecrscore_,
                                        threed = threed_,
                                        max3d = max3d_,
                                        output = output_,
                                        name = name_,
                                        output_no_images = output_no_images_,
                                        output_no_graph = output_no_graph_,
                                        display = display_,
                                        allow_tree = allow_tree_,
                                        max = max_,
                                        cutoff = cutoff_,
                                        radial = radial_,
                                        hub = hub_,
                                        fast = fast_,
                                        links_file = links_file_,
                                        known_actives_file = known_actives_file_,
                                        max_dist_from_actives = max_dist_from_actives_,
                                        )
        
        self.strict, self.loose = self.db_mol.build_matrices()
        if loadMatrix:
            self.sim_data = np.load(loadFile)
        else:
            self.sim_data = self.strict.to_numpy_2D_array()
            
        self.n_arr = _clean_NaN(self.sim_data)
        self.ID_List = _clean_ID_list(_getID(self.db_mol, self.n_arr))
        self.plotter = Plotter(self.ID_List)
        self.sub_arrs = None
        self.sub_IDs = None

    def saveDistanceMatrix(self, path : str):
        np.save(path, self.sim_data)

    def get_similarity_matrix(self, type : str):
        if type == "strict":
            return self.strict
        elif type == "loose":
            return self.loose
        else:
            raise ValueError("Invalid type. Please use 'strict' or 'loose'")
    
    def create_clusters(self):
        self.sub_arrs, self.sub_IDs = self.cluster_interactive()
        print(self.sub_arrs, self.sub_IDs)
        
    def getDBMolecules(self):
        '''
        Returns the lomap.DBMolecules object. Useful if one wants to access lomap functionalities
            
            Parameters:
                self: object, the ClusterMaker object.
            
            Returns:
                db_mol: object, the lomap.DBMolecules object.
        '''
        return self.db_mol
    
    def cluster_interactive(self):
        '''
        Function for if user wants to inspect distance data first and
        self assign the clustering neighbour distance cutoff. Also useful if
        testing different potential cutoff values. This is the current default.

            Parameters:
                self: object, the ClusterMaker object.

            Returns:
                sub_arrs: dict, n_arr subdivided into dict of cluster number keys
                sub_IDs: dict, ID_list subdivided into dict of cluster number keys
            
        '''
        def get_numeric(splitme):
            '''
            Will read str and return integer values.
            '''
            delimiters = [" ", ",", "[", "]", ")", "("]
            cleaned = splitme
            for i in delimiters:
                cleaned = cleaned.replace(i, " ")
            digs = [int(s) for s in cleaned.split() if s.isdigit()]
            return digs

        # Generate distance data
        self.data = 1 - self.sim_data

        # Output distance info to user.
        x, dists = _k_dist(self.data)
        auto_cutoff = _find_max_curvature(x, dists, savefigs=True, verbose=True)

        # Get user input on distance cutoff, epsilon in dbscan, typed
        input_cutoff = input("Enter a neighbor distance cutoff for clustering:")
        if input_cutoff == "":
            # If enter, set to autodetected cutoff
            user_cutoff = auto_cutoff
        else:
            user_cutoff = float(input_cutoff)
        # Do Clustering.
        # I changed the min_s from 4 to 2
        labels, mask, n_clusters_ = _dbscan(self.data, dist_cutoff=user_cutoff, min_s = 2)
        
        ax = self.plotter.plt_cluster(self.data, labels)
        
        # Output table for user cluster selection
        # Modify to just print clusters and create a sdf file useful for pyeds

        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)
        input("<Hit Enter>")
        plt.close()
        

        # Figure saving
        pdf = matplotlib.backends.backend_pdf.PdfPages("output.pdf")
        fig1 = self.plotter.plt_cluster(self.data, labels)
        fig2 = self.plotter.plt_dbscan(self.data, labels, mask, n_clusters_)
        pdf.savefig(fig1, bbox_inches= "tight", pad_inches=0.5)
        pdf.savefig(fig2)
        pdf.close()

        # Generate sub-arrays of clusters. Stored in dictionaries.
        sub_arr, sub_ID = _sub_arrays(labels, self.sim_data, self.ID_List)

        return sub_arr, sub_ID



    def clusters_w_ref(ref_ligs, sub_ID):
        '''
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


    def cluster_auto(self,data, ID_list, **kwargs):
        ''' The full automated sequence, not including outputting new arrays'''
        # Clean ID list if needed, func in utils.
        _clean_ID_list(ID_list)
        # Make output plots for PDF
        pdf = matplotlib.backends.backend_pdf.PdfPages("output.pdf")
        x, dists = _k_dist(data)
        epsilon_fit = _find_max_curvature(x, dists, savefigs=True)
        labels, mask, n_clusters_ = _dbscan(data)
        fig1 = self.plotter.plt_cluster(data, labels, ID_list)
        fig2 = self.plotter.plt_dbscan(data, labels, mask, n_clusters_)
        pdf.savefig(fig1)
        pdf.savefig(fig2)
        pdf.close()
        return labels
    
    def get_clusters(self, type : str, threshold : float):
        # TODO:
        pass


class Plotter():
    def __init__(self, id_list):
        self.ID_list = id_list

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
        cmap = cm.get_cmap('inferno', len(unique_labels))
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
        fig.tight_layout()
        plt.subplots_adjust(top=0.86, bottom=0.22)
        self.plt_cluster_regions(labels, ax=axs[0], fig=fig)
        self.plt_heatmap(data, axs[1], fig)
        axs[0].set_title('Cluster Regions', pad = 15)
        axs[1].set_title('Heatmap of chemical distance', pad = 15)
        return fig


   
        

if __name__ == "__main__":
    cmaker = ClusterMaker('../input/FreeSolv')
    cmaker.saveDistanceMatrix("distance_matrix_FullFreeSolv.npy")
    cmaker.create_clusters()

    