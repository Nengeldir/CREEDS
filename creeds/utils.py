import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, SpectralClustering
from kneed import KneeLocator, DataGenerator
import sys
import pandas as pd
import math
import os



def _clean_NaN(sim_np):
    '''
    If there are NaN values in the similarity matrix, replace them with 0

    Args:
        sim_np (np.array): similarity matrix as a numpy array
    Returns:
        n_arr (np.array): similarity matrix with NaN replaced with 0
    '''
    
    # Replace NaN with 0
    n_arr = np.nan_to_num(sim_np)
    return n_arr

def _getID(db_mol, n_arr):
    '''
    Generates ID lists from lomap.DBMolecules
    for clustering or other uses.

        Parameters:
            db_mol: object created by lomap.DBMolecules
            n_arr: chemical similarity 2D numpy array
        Returns:
            ID_list: list of ID names
    '''
    mol_names = ['ID']

    for i in range(n_arr.shape[1]):
        fname = db_mol[i].getName()
        fsplit = fname.split(".", 1)[0]
        mol_names.append(fsplit)

    # Convert the name list to an np array.
    mol_arr_pre = np.array(list(mol_names))
    # Cleave the 'ID' entry
    ID_list = mol_arr_pre[1:]

    return ID_list

def _clean_ID_list(ID_list):
    '''
    If ID_list (ligand names) assigned are only integer
    values, clean then up to add 'lig_' in the front. This is
    to avoid integer variable assignment during optimization.
    
        Parameters:
            ID_list: list of str
        Returns:
            ID_list_cleaned: list of str
        '''
    # String to add if str starts with int.
    add_str = str('lig_')
    # Start count.
    i = 0
    while i < len(ID_list):
    
        # replace str starting in int
        if ID_list[i][:1].isdigit() == True:
            ID_list[i] = f'{add_str}{ID_list[i]}'
        i += 1
        
    return ID_list

def _k_dist(data):
    '''
    Generates data for k-distance sorting to
    calculate the neighbor distance threshold for clustering.

        Parameters:
            data: distance array
        Returns:
            x: array the indexes each ligand
            distances: sorted nearest neighbor distances
    '''
    distances = np.sort(data, axis=0)
    distances = distances[:,1]
    x = np.linspace(0, distances.shape[0]-1, distances.shape[0])

    return x, distances

def _find_max_curvature(x, dists, plot_path='plots/', **kwargs):
    '''
    Outputs the point of maximum curvature, known as the elbow
    or knee of a curve.
        Parameters:
            x: x-values
            dists: distances from similarity array.
        Optional Parameters:
            savefigs: controls if figures are saved
            verbose: ouputs added prints to screen
        Returns:
            e_fit: the neighbor distance cutoff, calculated
                   or entered by user.
    '''
    # Measure concave/covex or increasing/decreasing.

    def detect_max(k, **kwargs):
        # Find max curvature point.
        e = k.knee_y
        # If starts flat, None may be detected. Fix.
        e = 0.0 if e is None else e
        epsilon = round(e, 3)
        k.plot_knee_normalized(figsize=(7,7))

        return epsilon
    
    dir, cv = _find_shape(x, dists)
    k_raw = KneeLocator(x, dists, S=1.0, curve=cv, direction=dir)
    k_fit = KneeLocator(x, dists, S=1.0, curve=cv, direction=dir,
                        interp_method="polynomial", online = True)
    
    try:
        e_raw = detect_max(k_raw)
    except:
        # If curve is not found, set neighbor dist to 0.5, user can override
        e_raw = 0.5
    
    plt.title("Raw Distance Data")
    plt.suptitle("Clustering cutoff at y-value of highest curvature",
                 size = 'medium', style = 'italic')
    plt.xlabel("Normalized ligands, sorted by distance")
    plt.ylabel("Cummulative distribution function of neighbor distance")
    plt.xticks(np.linspace(0.0, 1.0, 11, endpoint=True))
    plt.yticks(np.linspace(0.0, 1.0, 11, endpoint=True))
    cutoff_raw_path = os.path.join(plot_path, 'cutoff_raw.pdf')
    plt.savefig(cutoff_raw_path, pad_inches=0.3)
    
    if kwargs.get('verbose') is True:
        plt.show()
    
    try:
        e_fit = detect_max(k_fit)
    except:
        # If curve is not found, set neighbor dist to 0.5, user can override
        e_fit = 0.5
    plt.title("Polynomially Fit Distance Data")
    plt.suptitle("Clustering cutoff at y-value of highest curvature",
                 size = 'medium', style = 'italic')
    plt.xlabel("Normalized ligands, sorted by distance")
    plt.ylabel("Cummulative distribution function of neighbor distance")
    plt.xticks(np.linspace(0.0, 1.0, 11, endpoint=True))
    plt.yticks(np.linspace(0.0, 1.0, 11, endpoint=True))
    cutoff_fit_path = os.path.join(plot_path, 'cutoff_fit.pdf')
    plt.savefig(cutoff_fit_path, pad_inches=0.3)

    if kwargs.get('verbose') is True:
        plt.show()
        
    
    
    
    print(f"A suggested range for neighbor distances is between {e_raw} and {e_fit}.")
    print(f"The computed, default cutoff is {e_fit}.")
    
    
    # Testing block for analysis
    '''
    points = [0.05, 0.125, 0.175]

    for point in points:
    
        slope, dslope, eps_at_point = output_slopes(k_fit, point)
        print(f'The slope at {point} is {slope} for k_fit.')
        print(f'The slope of the diff curve at {point} is {dslope} for k_fit.')
        print(f'Epsilon at {point} = {eps_at_point}')
    
        slope2, dslope2, eps_at_point = output_slopes(k_raw, point)
        print(f'The slope at {point} is {slope2} for k_raw.')
        print(f'The slope of the diff curve at {point} is {dslope2} for k_raw.')
        print(f'Epsilon at {point} = {eps_at_point}')
    '''
    
    return e_fit

def get_num_neighbours(X, nneighbours):
    npoints = X.shape[0]
    nn_data = [[p, list(pd.Series(X[:, p]).sort_values().index[1:nneighbours])] for p in range(0, npoints)]
    nn_data = pd.DataFrame(nn_data, columns = ['index', 'nn']).set_index("index")
    return nn_data

def _cluster_dispersion(X, clusters):
    """
    Function that computes the cluster dispersion. The cluster dispersion is defined
    as the standard deviation in distance of all the elements within a cluster. The sum of the cluster dispersion
    of all the clusters in a dataset is called the total cluster dispersion. The lower the cluster dispersion,
    the more compact the clusters are.
    Inputs:
    X: numpy array of the distance matrix
    clusters: table with cluster labels of each event. columns: 'label', index: points
    """

    if "label" not in clusters.columns:
        raise ValueError("Table of clusters does not have 'label' column.")

    def std_distances(points, dm):
        distances = np.tril(dm[np.ix_(points, points)])
        distances[distances == 0] = np.nan
        cdispersion = np.nanstd(distances)
        return cdispersion

    nclusters = clusters["label"].nunique()
    points_per_cluster = [list(clusters[clusters.label == cluster].index) for cluster in range(nclusters)]
    wcsdist = [std_distances(points_per_cluster[cluster], X) for cluster in range(nclusters)]
    cluster_dispersion_df = pd.DataFrame(wcsdist, index=np.arange(nclusters), columns=["cdispersion"])
    return cluster_dispersion_df

def _get_cluster_neighbors(df, nn_df, nclusters, nneighbors):
    """
    Function to find the cluster neighbors of each cluster.
    The cluster neighbors are selected based on a smaller number of neighbours
    because I don't want to get no neighboring clusters.
    The minimun number of nn to get cluster neighbors is 30. This choice is arbitrary.
    Imputs:
        df: a table with points as index and a "label" column
    Returns:
        A dictionary of shape: {i: [neighbor clusters]}, i= 0,..,nclusters
    """
    def cluster_neighbor_for_point(nn_list, nneighbours):
        nn_labels = df1.loc[nn_list[0:nneighbours], 'label']
        return np.unique(nn_labels)

    df1 = df.copy()
    df1 = pd.merge(df1, nn_df, left_index=True, right_index=True)
    nn = min(30, nneighbors)  # nearest neighbours to compute border points: this def is arbitrary
    df1["unique_clusters"] = df1.apply(lambda row: cluster_neighbor_for_point(row["nn"], nn), axis=1)

    temp = df1[["unique_clusters"]]
    # get neighbor clusters (remove own cluster)
    neighbors = {}
    for c in range(nclusters):
        points_in_cluster = df1.label == c
        neighbors_in_cluster = temp.loc[points_in_cluster, "unique_clusters"].to_list()
        neighbors[c] = {i for l in neighbors_in_cluster for i in l if i != c}
    return neighbors

def _optimal_cluster_sizes(n_clusters, npoints):
    """
    Function to get the optimal cluster sizes. The optimal cluster sizes are defined as the number of points
    that each cluster should have in order to have the same number of points in each cluster. With the limitations
    of REEDS in OpenMM in mind.
    Inputs:
        n_clusters: number of clusters
        npoints: number of points
    Returns:
        optimal_cluster_sizes: list of the optimal cluster sizes
    """
    min_points = 3
    max_points = 28

    calc_min = math.floor(npoints / float(n_clusters))
    calc_max = math.floor(npoints / float(n_clusters)) + 1

    if calc_max > max_points:
        raise ValueError("The number of clusters is too high for REEDS in OpenMM")
    
    number_clusters_max = npoints % n_clusters
    number_cluster_min = n_clusters - number_clusters_max

    list1 = list(calc_max * np.ones(number_clusters_max, dtype=int))
    list2 = list(calc_min * np.ones(number_cluster_min, dtype=int))

    return list1 + list2

def _get_clusters_outside_range(clustering, min_range, max_range):
    """
    Function to get the clusters that are outside the range of the minimum and maximum number of points
    Inputs:
        clustering: table with the cluster labels of each event. columns: 'label', index: points
        min_range: minimum number of points in a cluster
        max_range: maximum number of points in a cluster
    Returns:
        large_clusters: list of clusters that are larger than the maximum number of points
        small_clusters: list of clusters that are smaller than the minimum number of points
    """
    
    cluster_sizes = clustering["label"].value_counts().reset_index()
    cluster_sizes.columns = ["cluster", "npoints"]
    
    large_clusters = list(cluster_sizes[cluster_sizes > max_range]["cluster"].dropna().values)
    small_clusters = list(cluster_sizes[cluster_sizes < min_range]["cluster"].dropna().values)
    

    return large_clusters, small_clusters

def _get_no_large_clusters(clustering, max_range):
    """
    Function to get clusters smaller than max_range
    Input: clustering: table with idx as points, and a "label" column
    """

    csizes = clustering.label.value_counts().reset_index()
    csizes.columns = ["cluster", "npoints"]

    return list(csizes[(csizes.npoints < max_range)]["cluster"].values)

def _get_points_to_switch(X, cl_elements, clusters_to_modify, idxc):
        """
        Function to obtain the closest distance of points in cl_elements with respect to the clusters in
        clusters_to_modify
        Inputs:
            X: distance matrix
            cl_elements: list of points of the cluster(s) that give points
            cluster_to_modify: a list of labels of clusters that receive points.
            idxc: dictionary with keys clusters_to_modify and values the points of these clusters, ex:
                  {'0': [idx1, idx2,...,idxn]}
        Returns:
            A table with the closest distance of points in clabel to clusters in
            clusters_to_modify
        """
        neighbor_cluster = []
        distances = []
        for point in cl_elements:
            dist = {c: X[idxc[c], point].mean() for c in clusters_to_modify}  # Instead of min. Worth future inv.
            new_label = min(dist, key=dist.get)  # closest cluster
            neighbor_cluster.append(new_label)
            distances.append(dist[new_label])

        cdistances = pd.DataFrame({"points": cl_elements, "neighbor_c": neighbor_cluster, "distance": distances})
        cdistances = cdistances.sort_values(by="distance", ascending=True).set_index("points")
        return cdistances

def _cluster_equalization(X, cluster_neighbours, n_clusters, equity_fraction, clustering):
    """
    Function to equalize the clusters. The function will try to make the clusters
    as equal as possible. The function will iteratively move points from the largest cluster
    to the smallest cluster until the difference in size between the largest and smallest cluster
    is smaller than a certain fraction of the total number of points.
    Inputs:
        X: numpy array of the distance matrix
        cluster_neighbours: dictionary with the cluster neighbours of each cluster
        equity_fraction: float (0, 1) to determine how equal the clusters should be. The higher the fraction,
            the more equal the clusters will be. The default is 0.1.
    Returns:
        labels: array of cluster numbers by ligand
        dispersion: the total cluster dispersion
        n_clusters_: the number of clusters
    """
    # get the number of points
    npoints = X.shape[0]
    
    # get optimal cluster sizes
    elements_per_cluster = _optimal_cluster_sizes(n_clusters, npoints)
    min_range = np.array(elements_per_cluster).min() * equity_fraction
    max_range = np.array(elements_per_cluster).max() * (2 - equity_fraction)

    range_points = (min_range, max_range)

    all_clusters = list(np.arange(0, n_clusters))
    clustering_c = clustering.copy()
    large_clusters, small_clusters = _get_clusters_outside_range(clustering_c, min_range, max_range)

    if (len(large_clusters) == 0) and (len(small_clusters) == 0):
        clustering_dispersion = _cluster_dispersion(X, clustering_c)
        return clustering_c, clustering_dispersion, clustering_dispersion.wcsd, n_clusters
    
    other_clusters = list(set(all_clusters) - set(large_clusters)) # clusters that receive points
    inx = {c: list(clustering_c[clustering_c.label == c].index) for c in other_clusters}

    for clarge in large_clusters: # make big cluters smaller
        cl_elements = list(clustering_c[clustering_c.label == clarge].index)
        
        closest_distance = _get_points_to_switch(X, cl_elements, other_clusters, inx)

        leftovers = len(cl_elements) - elements_per_cluster[clarge]

        for point in list(closest_distance.index):
            if leftovers <= 0:
                break

            new_label = closest_distance.loc[point, "neighbour_c"]
            points_new_label = clustering_c[clustering_c.label == new_label].shape[0]

            if points_new_label >= max_range:
                continue

            if new_label in cluster_neighbours[clarge]:
                clustering_c.at[point, "label"] = new_label
                inx[new_label].append(point)
                leftovers -= 1
        
        other_clusters = _get_no_large_clusters(clustering_c, max_range)
        inx = {c: list(clustering_c[clustering_c.label == c].index) for c in other_clusters}

    large_clusters = _get_no_large_clusters(clustering_c, max_range)
    clusters_to_steal = list(set(all_clusters) - set(large_clusters)) # clusters that give points

    if len(small_clusters) == 0:
        clustering_dispersion = _cluster_dispersion(X, clustering_c)
        return clustering_c, clustering_dispersion, n_clusters
    else: # get bigger the small clusters
        cl_elements = list(clustering_c[clustering_c.label.isin(clusters_to_steal)].index)
        inx = {c: list(clustering_c[clustering_c.label == c].index) for c in small_clusters}
        closest_distance = _get_points_to_switch(X, cl_elements, large_clusters, inx)

        needed_points = {c: min_range - clustering_c[clustering_c.label == c].shape[0] for c in small_clusters}

        for point in list(closest_distance.index):
            new_label = closest_distance.loc[point, "neighbour_c"]
            current_label = clustering_c.loc[point, "label"]
            points_current_label = clustering_c[clustering_c.label == current_label].shape[0]

            if needed_points[new_label] <= 0:
                break
            
            if points_current_label <= min_range:
                continue

            if new_label in cluster_neighbours[current_label]:
                clustering_c.at[point, "label"] = new_label
                needed_points[new_label] -= 1

        clustering_dispersion = _cluster_dispersion(X, clustering_c)
        return clustering_c, clustering_dispersion, n_clusters
    
def _spectral(X, num_clusters, **kwargs):
    '''
    Peforms clustering using Spectral Clustering.
    
        Parameters:
            X: 2D distance matrix from similarity.
            n_clusters: number of clusters to find
        Optional Parameters:
            n_init: number of times the k-means algorithm will be run
                    with different centroid seeds
                    default = 10
            equity_fraction: float (0, 1) to determine how equal the clusters should be. The higher the fraction,
                the more equal the clusters will be. The default is 0.1.
            nneighbours: int, number of neighbours to consider used to construct affinity matrix
            seed: int, random seed
        Returns:
            labels: array of cluster numbers by ligand
            core_samples_mask: filters clusters
            n_clusters_: the number of clusters
    '''
    # Define optional arugments
    # Number of times the k-means algorithm will be run
    n_init = kwargs.get('n_init', 10)
    equity_fraction = kwargs.get('equity_fraction', 0.1)
    nneighbours = kwargs.get('nneighbours', 10)
    seed = kwargs.get('seed', None)

    # Find the number of neighbours for each ligand 
    # Note that the ligand itself is not a neighbour to itself.
    nn_df = get_num_neighbours(X, nneighbours)

    # Find intial clustering.
    sc1 = SpectralClustering(n_clusters = num_clusters,
                            affinity = 'precomputed_nearest_neighbors',
                            random_state = seed,
                            n_neighbors = nneighbours,
                            n_init = n_init).fit(X)
    labels = sc1.labels_
    first_clustering = pd.DataFrame(labels, columns = ["label"])
    #calculate dispersion of clusters
    dispersion1 = _cluster_dispersion(X, first_clustering)
    total_dispersion1 = dispersion1["cdispersion"].sum()
    print(f"Total dispersion of first spectral clustering: {total_dispersion1}")

    # get cluster neighbours
    cluster_neighbours = _get_cluster_neighbors(first_clustering, nn_df, num_clusters, nneighbours)

    # equalize clusters
    labels, dispersion, n_clusters_ = _cluster_equalization(X, cluster_neighbours, num_clusters, equity_fraction, first_clustering)
    
    # convert labels to np array
    labels = labels["label"].values

    print(f"Total dispersion of final spectral clustering: {dispersion['cdispersion'].sum()}")

    return labels, n_clusters_

def _dbscan(X, max_searches = 10, plot_path = 'plots/', **kwargs):
    '''
    Peforms clustering using DBSCAN.
    
        Parameters:
            X: 2D distance matrix from similarity.
        Optional Parameters:
            dist_cutoff: neighbor distance cutoff
                         default = None, will calculate
            min_s: minimum sample in cluster
                   default = 1
            max_searches: number of maximal searches to perform to find a good fit
                    default = 10
        Returns:
            labels: array of cluster numbers by ligand
            core_samples_mask: filters clusters
            n_clusters_: the number of clusters
    '''
    # Define optional arugments
    # Minimum sample in cluster
    min_s = kwargs.get('min_s', 1)
    if min_s is None:
        min_s = 1
    else:
        min_s = min_s
    # Max distances apart to be neighbors
    dist_cutoff = kwargs.get('dist_cutoff', None)
    if dist_cutoff is None:
        # If not given, calculate it. Default.
        x, dists = _k_dist(X)
        dist_cutoff = _find_max_curvature(x, dists, plot_path)
    else:
        dist_cutoff = dist_cutoff
    

    # Find clusters.
    db = DBSCAN(eps=dist_cutoff, min_samples=min_s, metric = 'precomputed').fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Find number of clusters, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    # Print cluster information for user.
    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

    # Ask user if they want to fit the noise. and write down ligands 
    # that are noise and to which cluster they are mapped
    fit_noise = input("Do you want to try to fit the noise?")
    if fit_noise:
        print("Adding the noise ligands to the cluster with the closest ligand")
        for i, label in enumerate(labels):
            if label == -1:
                distances = X[i]
                sorted_distances = np.sort(distances)

                # maximum number of times one can search for a possible fit for the outlier
                # that is again not an outlier
                for j in range(max_searches): 
    
                    # get the index of the ligand that is most similar
                    # as the matrix is symmetric we can skip the same value 
                    loc_min_dist = sorted_distances[2 * j] 
                    loc_min_index = np.where(distances == loc_min_dist)[0][0]

                    #the closest match is also an outlier
                    if labels[loc_min_index] != -1:
                        label_closest_match = labels[loc_min_index]
                        labels[i] = label_closest_match        
                        break
                    
                    # we have not found a good match for the ligand: Throw an exception
                    if j == max_searches - 1:
                        raise ValueError("For ligand " +  str(i) + " no good fit was found after " + str(max_searches) + " searches")
    else:
        print("losing %d ligands", n_noise_)

    return labels, core_samples_mask, n_clusters_

def _find_shape(x, y):
    """
    Detect the direction and curvature of k-dist line.
        
        Parameters:
            x, y: x and y values
        Returns:
            direction: "increasing" or "decreasing"
            curve type: "concave" or "convex"
    """
    p = np.polyfit(x, y, deg=1)
    x1, x2 = int(len(x) * 0.2), int(len(x) * 0.8)
    q = np.mean(y[x1:x2]) - np.mean(x[x1:x2] * p[0] + p[1])
    if p[0] > 0 and q > 0:
        return 'increasing', 'concave'
    if p[0] > 0 and q <= 0:
        return 'increasing', 'convex'
    if p[0] <= 0 and q > 0:
        return 'decreasing', 'concave'
    else:
        return 'decreasing', 'convex'

def _sub_arrays(labels, n_arr, ID_list):
    '''
    Make dictionaries containing clusters to optimize. Keys are cluster
    numbers. Key contents are np similarity arrays (sub_arrs)
    or ligand names (sub_IDs).
    
        Parameters:
            labels: cluster numbers calculated by DBSCAN()
            n_arr: the 2D similarity array
            ID_list: list of ID names
             
        Returns:
            sub_arrs: dict, n_arr subdivided into dict of cluster number keys
            sub_IDs: dict, ID_list subdivided into dict of cluster number keys
    '''
    # Clean ID list if needed, func in utils.
    _clean_ID_list(ID_list)
    
    # Filter out ligs measured as noise. Noise is cluster -1.
    labels_w_o_noise = [x for x in labels if x >= 0]
    unique_labels = set(labels_w_o_noise)
    
    # Generate dictionary of names for submatrices, named by cluster index (0,...N)
    sub_arrs = dict((i, "n_arr_" + str(i)) for i in range(len(unique_labels)))
    sub_IDs = dict((i, "IDs_" + str(i)) for i in range(len(unique_labels)))
    # Loop over the unique labels to generate similarity matrices of clusters
    c = 0
    for i in unique_labels:
        # Find label entries corresponding to clusters
        result = np.where(labels == i)
        # Ouput dict of clustered similarity np arrays
        sub_arrs[c] = n_arr[np.ix_(result[0][:],result[0][:])]
        # Ouput dict of clustered ID lists
        sub_IDs[c] = [ID_list[index] for index in result[0][:]]
        c = c + 1
    return sub_arrs, sub_IDs