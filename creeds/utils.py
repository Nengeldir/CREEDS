import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from kneed import KneeLocator, DataGenerator

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

def _find_max_curvature(x, dists, **kwargs):
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
    plt.ylabel("Normalized nearest neighbor distance")
    plt.savefig('plots/cutoff_raw.pdf', pad_inches=0.3)
    
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
    plt.ylabel("Normalized nearest neighbor distance")
    plt.savefig('plots/cutoff_fit.pdf', pad_inches=0.3)

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

def _dbscan(X, **kwargs):
    '''
    Peforms clustering using DBSCAN.
    
        Parameters:
            X: 2D distance matrix from similarity.
        Optional Parameters:
            dist_cutoff: neighbor distance cutoff
                         default = None, will calculate
            min_s: minimum sample in cluster
                   default = 1
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
        dist_cutoff = _find_max_curvature(x, dists)
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