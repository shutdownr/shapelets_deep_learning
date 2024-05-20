import numpy as np
import torch

from sklearn.cluster import KMeans

# Multivariate time series shapelet extraction function
def get_shapelets_mts(X:np.ndarray, l:int, s:int=1, return_channels:bool=False, y:np.ndarray=None, return_labels:bool=False):
    """
    `X` is an ndarray with the shape (`N`, `C`, `L`), where `N` is the
    number of samples, `C` is the number of input channels, and `L` is the length of
    the input.

    `l` is an integer, representing the shapelet length to be extracted.

    `s` is an integer, representing the sliding window step size (stride)

    `return_channels` is a bool, if true the associated channels of each shapelet
    are returned

    `return_labels` is a bool, if true the associated labels of each shapelet
    are returned

    Outputs a 2D ndarray of shapelets (`S`, `l`), where
    S=N*C*(L-l+1) is the number of shapelets and l is the shapelet length.

    Optionally outputs a 1D ndarray of associated channels (`S`)
    
    Optionally outputs a 1D ndarray of associated labels (`S`)

    Note: This function assumes that all input samples have the same number of
    channels and the same length.
    """
    shapelets = np.lib.stride_tricks.sliding_window_view(X, window_shape=l, axis=2)
    out = [shapelets.reshape(-1, shapelets.shape[-1])[::s]]
    if return_channels:
        shapelet_channels = np.tile(np.arange(X.shape[1]), X.shape[2]-l+1).reshape(-1,X.shape[1]).T.flatten()
        shapelet_channels = np.tile(shapelet_channels, X.shape[0])
        out.append(shapelet_channels[::s])
    if return_labels:
        shapelet_labels = np.tile(y, (shapelets.shape[2], shapelets.shape[1], 1)).T.flatten()
        out.append(shapelet_labels[::s])
    return tuple(out) if len(out) > 1 else out[0]

def shapelet_discovery_mts(X:np.ndarray, encoder:torch.nn.Module, num_clusters:int):
    """
    `X` is an ndarray with the shape (`N`, `C`, `L`), where `N` is the
    number of samples, `C` is the number of input channels, and `L` is the length of
    the input.

    `y` is an ndarray with the shape (`N`)

    `encoder` is a pretrained pytorch Module, trained to encode similar shapelets close
    together in latent space

    `num_clusters` is an integer indicating the number of clusters used for KMeans
    
    Outputs a tuple with
        (1), a 2D ndarray of shapelets (`S`, `l`), and
        (2), a 1D ndarray of channels (`S`) respective to each shapelet, where

    S=N*C*(L-l+1) is the number of shapelets and l is the shapelet length.
    """
    beta = 0.5

    # TODO: Maybe use multiple shapelet lengths?
    l = 5

    # Get shapelets, with their respective channels
    shapelets, channels = get_shapelets_mts(X, l=l, return_channels=True)
    # Need to add a dimension for the channel to have a tensor of shape (`N`, 1, `l`)
    encoder.eval()
    with torch.no_grad():
        encoded_shapelets = encoder(torch.tensor(shapelets).unsqueeze(1))
    encoder.train()

    # Flatten (old code for different shapelet lengths)
    # shapelets = [item for sublist in list(shapelets) for item in sublist]
    # encoded_shapelets = np.concatenate(encoded_shapelets)
    # channels = np.concatenate(channels)
    # labels = np.concatenate(labels)

    # Cluster the encoded shapelets with the number of clusters as passed
    kmeans = KMeans(n_clusters=num_clusters, n_init="auto")
    kmeans.fit(encoded_shapelets)

    # Shapelet candidates and their channels
    shapelet_candidates = []
    shapelet_channels = []
    # For utility calculation
    cluster_sizes = []
    candidates_encoded = []

    for i in range(num_clusters):
        # Relevant indices for cluster i
        cluster_indices = kmeans.labels_==i
        # Original indices for the selected shapelets
        shapelet_indices = np.arange(len(encoded_shapelets))[cluster_indices]

        # Find the shapelet with minimum distance to the cluster center
        distances = [np.linalg.norm(q - kmeans.cluster_centers_[i]) for q in encoded_shapelets[cluster_indices]]
        min_index = np.argmin(distances)

        # The found shapelet with the closest distance to the cluster center
        shapelet_candidates.append(shapelets[shapelet_indices[min_index]])
        shapelet_channels.append(channels[shapelet_indices[min_index]])

        # Number of shapelets assigned to cluster i
        cluster_sizes.append(np.sum(cluster_indices))
        candidates_encoded.append(encoded_shapelets[shapelet_indices[min_index]])

    # Get the shapelets sorted by utility
    # Sort shapelet candidates by utility (only relevant if we have more than 1 cluster)
    if num_clusters > 1:
        # For term 1 (maximum cluster size)
        max_cluster_size = np.max(cluster_sizes)
        # For term 2 (sum of normalized distances between encoded candidate and other encoded candidates)
        normalized_distance_sums = [np.sum([np.linalg.norm(candidates_encoded[i] - c) for c in candidates_encoded]) for i in range(num_clusters)]
        max_normalized_distance_sum = np.max(normalized_distance_sums)
        # Utility calculation
        utility = []
        for i in range(num_clusters):
            utility_term1 = beta * np.log(cluster_sizes[i]) / np.log(max_cluster_size)
            utility_term2 = (1-beta) * np.log(normalized_distance_sums[i]) / np.log(max_normalized_distance_sum)
            utility.append(utility_term1 + utility_term2)

        utility = np.array(utility)

        sort_order = np.argsort(-utility)
        shapelet_candidates = np.array(shapelet_candidates,dtype=object)[sort_order]
        shapelet_channels = np.array(shapelet_channels)[sort_order]

    return shapelet_candidates, shapelet_channels

# def _encode_shapelets(shapelets:torch.tensor, encoder:torch.nn.Module):
#     """
#     `shapelets` is a tensor with the shape (`N`, `L`), where `N` is the
#     number of shapelets, and `L` is the length of each shapelet.

#     `y` is an ndarray with the shape (`N`)

#     Outputs a 2D ndarray of encoded shapelets (`N`, `d`), where `d`is the
#     dimensionality of the latent space
#     """
#     # Encode all shapelets
#     encoder.eval()
#     with torch.no_grad():
#         encoded_shapelets = encoder(shapelets)
#     encoder.train()

#     return encoded_shapelets