import numpy as np

def shapelet_transform_mts(X:np.ndarray, shapelets:np.ndarray, channels:np.ndarray):
    """
    Applies multivariate shapelet transform on an input dataset X 
    based on given shapelets and their associated channels

    `X` is an ndarray with the shape (`N`, `C`, `L`), where `N` is the
    number of samples, `C` is the number of input channels, and `L` is the length of
    the input.

    `shapelets` is a 2D ndarray with the shape (`m`, `l`),

    `channels` is a 1D ndarray with associated channels with the shape (`m`)

    Outputs a 2D ndarray of features with the shape (`N`, `m`), where
    m is the reduced dimensionality, based on the number of shapelets `m`
    """
    features = []

    # For each sample
    for sample in X:
        feature = []
        # For each discovered shapelet candidate
        for shapelet, channel in zip(shapelets, channels):
            min_dist = np.inf
            s_length = len(shapelet)
            for k in range(X.shape[2]-s_length+1):
                distance = sample[channel, k:k+s_length] - shapelet.astype(float)
                distance_norm = np.linalg.norm(distance)
                if distance_norm < min_dist:
                    min_dist = distance_norm
            feature.append(min_dist)
        features.append(feature)
    return np.array(features)

def shapelet_transform_text(X:np.ndarray, shapelets:np.ndarray):
    # TODO: Implement
    pass