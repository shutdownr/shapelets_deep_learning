import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
import torch

from resources.encoder import CausalCNNEncoder
from resources.text import Tokenizer
from shapelet_extraction import shapelet_discovery_mts, shapelet_discovery_text
from shapelet_transform import shapelet_transform_mts, shapelet_transform_text
from triplet_loss import PNTripletLossMTS, PNTripletLossText
from utils import Dataset

def get_default_config():
    """
    Outputs the default configuration for training as a dict
    """
    return {
        # Model parameters

        # CNN parameters
        # Input channels is always 1 (since we are using 1D convolutions)
        "in_channels": 1,
        # Hidden channels within the CNN layers
        "channels": 20,
        "depth": 3,
        # Output size of the convolutional layers
        "reduced_size": 80,
        # Convolution kernel size
        "kernel_size": 3,
        # Output dimensionality of the encoder
        "out_channels": 160,

        # Stride for shapelet extraction
        "stride": 1,
        # Shapelet length
        "l": [5],

        # Training parameters
        "batch_size": 16,
        "epochs": 30,
        "lr": 0.001,

        # Number of shapelets (based on final clustering)
        "num_clusters": 25,
        # Final classifier
        "classifier": SVC(kernel="linear", gamma="auto")

    }

def train_mts(X_train:np.array, config:dict, random_state:int=42, debug:bool=False):
    """
    Trains a CausalCNNEncoder based on the PNTripletLoss with `X_train`and the
    `config` as passed

    `X_train` is an ndarray with the shape (`N`, `C`, `L`), where `N` is the
    number of samples, `C` is the number of input channels, and `L` is the length of
    the input.

    `config` is a dict containing configuration parameters for the model and training

    `random_state` is an int, representing the random seed 

    `debug` is a bool flag, responsible for debug prints
    
    Outputs a tuple with
        - (1), a 1D ndarray of the history of training losses per epoch 
        - (2), a CausalCNNEncoder, the trained encoder network 
    """
    train_dataset = Dataset(X_train)
        
    # TODO: Make this work for ragged input data
    train_generator = torch.utils.data.DataLoader(train_dataset, config["batch_size"], collate_fn=None, shuffle=True)

    loss_function = PNTripletLossMTS(config)
    encoder = CausalCNNEncoder(config["in_channels"],
                        config["channels"],
                        config["depth"],
                        config["reduced_size"],
                        config["out_channels"],
                        config["kernel_size"]).double()
    
    optimizer = torch.optim.Adam(encoder.parameters(), lr=config["lr"])

    # Encoder training
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    history = []
    for i in range(config["epochs"]):
        for batch in train_generator:
            optimizer.zero_grad()
            # No model call here, that is done in the loss function directly
            loss = loss_function(batch, encoder)
            loss.backward()
            optimizer.step()
        if debug:
            print("Epoch", i+1, loss.detach().numpy()[0])
        history.append(loss.detach().numpy()[0])
    history = np.array(history)
    return history, encoder

def classify_shapelets_mts(X_train:np.ndarray, y_train:np.ndarray, X_test:np.ndarray, y_test:np.ndarray, config:dict, encoder:torch.nn.Module):
    # Get the best shapelets from the train data
    shapelets, channels = shapelet_discovery_mts(X_train, encoder, config)

    # Shapelet transform on train and test data
    features_train = shapelet_transform_mts(X_train, shapelets, channels)
    features_test = shapelet_transform_mts(X_test, shapelets, channels)

    # Fit the simple classifier on the train data
    classifier = config["classifier"]
    classifier.fit(features_train, y_train)
    # Evaluate the classifier on the test data
    accuracy = classifier.score(features_test, y_test)
    print("Accuracy:", accuracy)


def train_text(X_train:np.array, config:dict, random_state:int=42, debug:bool=False):
    """
    Trains a CausalCNNEncoder based on the PNTripletLoss with `X_train`and the
    `config` as passed

    `X_train` is an ndarray with the shape (`N`, `C`, `L`), where `N` is the
    number of samples, `C` is the number of input channels, and `L` is the length of
    the input.

    `config` is a dict containing configuration parameters for the model and training

    `random_state` is an int, representing the random seed 

    `debug` is a bool flag, responsible for debug prints
    
    Outputs a tuple with
        - (1), a 1D ndarray of the history of training losses per epoch 
        - (2), a CausalCNNEncoder, the trained encoder network 
        - (3), a Tokenizer, trained on the data 
    """
    tokenizer = Tokenizer()
    tokenizer.fit(X_train)

    train_dataset = Dataset(X_train)
        
    train_generator = torch.utils.data.DataLoader(train_dataset, config["batch_size"], collate_fn=None, shuffle=True)

    loss_function = PNTripletLossText(config, tokenizer)
    encoder = CausalCNNEncoder(config["in_channels"],
                        config["channels"],
                        config["depth"],
                        config["reduced_size"],
                        config["out_channels"],
                        config["kernel_size"]).double()
    
    optimizer = torch.optim.Adam(encoder.parameters(), lr=config["lr"])

    # # Encoder training
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    history = []
    for i in range(config["epochs"]):
        for batch in train_generator:
            optimizer.zero_grad()
            # No model call here, that is done in the loss function directly
            loss = loss_function(batch, encoder)
            loss.backward()
            optimizer.step()
        if debug:
            print("Epoch", i+1, loss.detach().numpy()[0])
        history.append(loss.detach().numpy()[0])
    history = np.array(history)
    return history, encoder, tokenizer

def classify_shapelets_text(X_train:np.ndarray, y_train:np.ndarray, X_test:np.ndarray, y_test:np.ndarray, config:dict, encoder:torch.nn.Module):
    # Get the best shapelets from the train data
    shapelets = shapelet_discovery_text(X_train, encoder, config)

    # Shapelet transform on train and test data
    features_train = shapelet_transform_text(X_train, shapelets)
    features_test = shapelet_transform_text(X_test, shapelets)

    # Fit the simple classifier on the train data
    classifier = config["classifier"]
    classifier.fit(features_train, y_train)
    
    # Evaluate the classifier on the test data
    accuracy = classifier.score(features_test, y_test)
    print("Accuracy:", accuracy)

def plot_history(history:np.ndarray, filename:str=None):
    """
    Plots a history of training losses

    `history` is a 1D ndarray of training losses
    """
    plt.figure(figsize=(12,6))
    plt.plot(history)
    plt.xticks(np.arange(len(history)))
    plt.xlabel("Epoch")
    plt.ylabel("Triplet Loss")

    if filename:
        plt.savefig(filename)
