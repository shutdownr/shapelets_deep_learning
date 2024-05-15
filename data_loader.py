from aeon.datasets.dataset_collections import multivariate
from aeon.datasets import load_classification

import numpy as np
import pandas as pd

class DataLoader:
    def __init__(self):
        self.mts_datasets = None

    def get_mts_datasets(self):
        if self.mts_datasets is None:
            working_datasets = []
            for dataset in list(multivariate):
                # Most of the datasets throw an exception when trying to load them
                # Try loading all and store the ones that work
                try: 
                    _ = self.load_mts_dataset(dataset)
                    working_datasets.append(dataset)
                except:
                    pass
            self.mts_datasets = working_datasets
        return self.mts_datasets

    def load_mts_dataset(self, dataset_name):
        X, y = load_classification(dataset_name)
        _, y = np.unique(y, return_inverse=True)
        return X, y

    def load_text_dataset(self, data_path):
        df = pd.read_csv(f"{data_path}/IMDB Dataset.csv")
        X = df["review"].to_numpy()
        _, y = np.unique(df["sentiment"].to_numpy(), return_inverse=True)
        return X, y