import numpy as np
import torch

from difflib import SequenceMatcher
from sklearn.cluster import AgglomerativeClustering, KMeans

from .text import Tokenizer, EncoderBPE

from typing import List

# Multivariate time series shapelet extraction function
def get_shapelets_mts(X:np.ndarray, l:int, s:int=1, return_channels:bool=False, Y:np.ndarray=None, return_labels:bool=False):
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
    # Fix for ragged MTS passed as list
    if type(X) == list:
        return get_shapelets_ragged_mts(X, l, s, return_channels, Y, return_labels)
    
    shapelets = np.lib.stride_tricks.sliding_window_view(X, window_shape=l, axis=2)
    out = [shapelets.reshape(-1, shapelets.shape[-1])[::s]]
    if return_channels:
        shapelet_channels = np.tile(np.arange(X.shape[1]), X.shape[2]-l+1).reshape(-1,X.shape[1]).T.flatten()
        shapelet_channels = np.tile(shapelet_channels, X.shape[0])
        out.append(shapelet_channels[::s])
    if return_labels:
        shapelet_labels = np.tile(Y, (shapelets.shape[2], shapelets.shape[1], 1)).T.flatten()
        out.append(shapelet_labels[::s])
    return tuple(out) if len(out) > 1 else out[0]

def get_shapelets_ragged_mts(X:list, l:int, s:int=1, return_channels:bool=False, Y:np.ndarray=None, return_labels:bool=False):
    all_shapelets = []
    all_channels = []
    all_labels = []
    for x,y in zip(X,Y):
        shapelets = np.lib.stride_tricks.sliding_window_view(x, window_shape=l, axis=1)
        shapelets = shapelets.reshape(-1, shapelets.shape[-1])
        all_shapelets.extend(shapelets)
        if return_channels:
            channels = np.tile(np.arange(x.shape[0]), x.shape[1]-l+1).reshape(-1,x.shape[0]).T.flatten()
            all_channels.extend(channels)
        if return_labels:
            all_labels.extend(np.tile(y,shapelets.shape[0]))
    out = [np.array(all_shapelets)[::s]]
    if return_channels:
        out.append(np.array(all_channels)[::s])
    if return_labels:
        out.append(np.array(all_labels)[::s])
    return tuple(out) if len(out) > 1 else out[0]

def shapelet_discovery_mts(X:np.ndarray, encoder:torch.nn.Module, config:dict):
    """
    `X` is an ndarray with the shape (`N`, `C`, `L`), where `N` is the
    number of samples, `C` is the number of input channels, and `L` is the length of
    the input.

    `encoder` is a pretrained pytorch Module, trained to encode similar shapelets close
    together in latent space

    `config` is the configuration dictionary
    
    Outputs a tuple with
        - (1), a 2D ndarray of shapelets (`S`, `l`), and
        - (2), a 1D ndarray of channels (`S`) respective to each shapelet, where

    S=N*C*(L-l+1) is the number of shapelets and l is the shapelet length.
    """
    beta = 0.5

    # TODO: Maybe use multiple shapelet lengths?
    # Get shapelets, with their respective channels
    shapelets, channels = get_shapelets_mts(X, l=config["l"][0], s=config["stride"], return_channels=True)
    # Need to add a dimension for the channel to have a tensor of shape (`N`, 1, `l`)
    encoder.eval()
    with torch.no_grad():
        encoded_shapelets = encoder(torch.tensor(shapelets))
    encoder.train()

    # Flatten (old code for different shapelet lengths)
    # shapelets = [item for sublist in list(shapelets) for item in sublist]
    # encoded_shapelets = np.concatenate(encoded_shapelets)
    # channels = np.concatenate(channels)
    # labels = np.concatenate(labels)

    # Cluster the encoded shapelets with the number of clusters as passed
    kmeans = KMeans(n_clusters=config["num_clusters"], n_init="auto")
    kmeans.fit(encoded_shapelets)

    # Shapelet candidates and their channels
    shapelet_candidates = []
    shapelet_channels = []
    # For utility calculation
    cluster_sizes = []
    candidates_encoded = []

    for i in range(config["num_clusters"]):
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
    if config["num_clusters"] > 1:
        # For term 1 (maximum cluster size)
        max_cluster_size = np.max(cluster_sizes)
        # For term 2 (sum of normalized distances between encoded candidate and other encoded candidates)
        normalized_distance_sums = [np.sum([np.linalg.norm(candidates_encoded[i] - c) for c in candidates_encoded]) for i in range(config["num_clusters"])]
        max_normalized_distance_sum = np.max(normalized_distance_sums)
        # Utility calculation
        utility = []
        for i in range(config["num_clusters"]):
            utility_term1 = beta * np.log(cluster_sizes[i]) / np.log(max_cluster_size)
            utility_term2 = (1-beta) * np.log(normalized_distance_sums[i]) / np.log(max_normalized_distance_sum)
            utility.append(utility_term1 + utility_term2)

        utility = np.array(utility)

        sort_order = np.argsort(-utility)
        shapelet_candidates = np.array(shapelet_candidates,dtype=object)[sort_order]
        shapelet_channels = np.array(shapelet_channels)[sort_order]

    return shapelet_candidates, shapelet_channels

def shapelet_discovery_text(X:List[str], tokenizer:Tokenizer, encoder:torch.nn.Module, config:dict):
    """
    `X` is a list of texts

    `y` is an ndarray with the shape (`N`)

    `encoder` is a pretrained pytorch Module, trained to encode similar shapelets close
    together in latent space

    `config` is the configuration dictionary
    
    Outputs a tuple with
        - (1), a 2D ndarray of shapelets (`S`, `l`), and
        - (2), a 1D ndarray of channels (`S`) respective to each shapelet, where

    S=N*C*(L-l+1) is the number of shapelets and l is the shapelet length.
    """
    beta = 0.5

    # TODO: Implement
    # See shapelet discovery mts

    # 1. get all shapelets from X
    e = EncoderBPE(tokenizer)
    e.fit(X)
    shapelets = e.get_filtered_shapelets(pad=True, **config)

    # 2. encode all shapelets with the trained encoder
    encoder.eval()
    with torch.no_grad():
        encoded_shapelets = encoder(torch.tensor(shapelets))
    encoder.train()

    # 3. based on config["num_cluster"] get n clusters of all encoded shapelets
    similarity_matrix = np.eye(len(shapelets))
    for i, s1 in enumerate(shapelets):
        s1 = s1[s1>0]
        for j, s2 in enumerate(shapelets[:i]):
            s = SequenceMatcher(None, s1, s2[s2>0]).ratio()
            similarity_matrix[i,j] = s
            similarity_matrix[j,i] = s

    clusters = AgglomerativeClustering(n_clusters=config["num_clusters"], metric="precomputed", linkage="complete").fit(1-similarity_matrix)
    cluster_labels = clusters.labels_

    # 4. for each cluster get the shapelet closest to the cluster center

    # Shapelet candidates and their channels
    shapelet_candidates = []
    # For utility calculation
    cluster_sizes = []
    candidates_encoded = []

    for i in range(config["num_clusters"]):
        # get intra-cluster-similarities: 
        cluster_mask = cluster_labels == i
        cluster_similarities = (1.-similarity_matrix)[cluster_mask,:][:,cluster_mask]

        # squared similarities penalize outliers:
        cluster_similarities *= cluster_similarities

        # get medoid (i.e. sample with smallest mean distance to all others):
        cluster_medoid = np.argmin(cluster_similarities.mean(axis=1))
        cluster_medoid = np.arange(len(shapelets))[cluster_mask][cluster_medoid]
        shapelet_candidates.append(shapelets[cluster_medoid])

        # Number of shapelets assigned to cluster i
        cluster_sizes.append(np.sum(cluster_mask))
        candidates_encoded.append(encoded_shapelets[cluster_medoid])

    # 5. sort shapelet by utility (I think that's useless, only relevant when using multiple shapelets per cluster)
    #       => the utility calculation can be done in exactly the same way as for MTS
    if config["num_clusters"] > 1:
        # For term 1 (maximum cluster size)
        max_cluster_size = np.max(cluster_sizes)
        # For term 2 (sum of normalized distances between encoded candidate and other encoded candidates)
        normalized_distance_sums = [np.sum([np.linalg.norm(candidates_encoded[i] - c) for c in candidates_encoded]) for i in range(config["num_clusters"])]
        max_normalized_distance_sum = np.max(normalized_distance_sums)
        # Utility calculation
        utility = []
        for i in range(config["num_clusters"]):
            utility_term1 = beta * np.log(cluster_sizes[i]) / np.log(max_cluster_size)
            utility_term2 = (1-beta) * np.log(normalized_distance_sums[i]) / np.log(max_normalized_distance_sum)
            utility.append(utility_term1 + utility_term2)

        utility = np.array(utility)

        sort_order = np.argsort(-utility)
        shapelet_candidates = np.array(shapelet_candidates,dtype=object)[sort_order]

    # 6. returns a list of shapelets (or ndarray)
    return shapelet_candidates