import numpy as np

def get_labeled_shapelets_by_channel(X:np.ndarray, y:np.ndarray, l:int, s:int=1):
    """
    Extracts shapelets of length l from an input dataset X with labels y 
    using a sliding window approach with a step size of s

    `X` is an ndarray with the shape (`N`, `C`, `L`), where `N` is the
    number of samples, `C` is the number of input channels, and `L` is the length of
    the input.

    `y` is an ndarray with the shape (`N`, `1`) where `N` is the number
    of samples, with the second dimension representing the class label.

    `l` is an integer, representing the shapelet length to be extracted.

    `s` is an integer, representing the sliding window step size (stride)

    Outputs a tuple with
        - (1), a 2D ndarray of shapelets (`S`, `l`),
        - (2), a 1D ndarray of associated channels (`S`), and
        - (3), a 1D ndarray of associated labels (`S`), where

    S=N*C*(L-l+1) is the number of shapelets and l is the shapelet length.

    Note: This function assumes that all input samples have the same number of
    channels and the same length.
    """
    shapelets = np.lib.stride_tricks.sliding_window_view(X, window_shape=l, axis=2)

    shapelet_labels = np.tile(y, (shapelets.shape[2], shapelets.shape[1], 1)).T.flatten()

    shapelet_channels = np.tile(np.arange(X.shape[1]), X.shape[2]-l+1).reshape(-1,X.shape[1]).T.flatten()
    shapelet_channels = np.tile(shapelet_channels, X.shape[0])

    shapelets = shapelets.reshape(-1, shapelets.shape[-1])
    return shapelets[::s], shapelet_channels[::s], shapelet_labels[::s]
