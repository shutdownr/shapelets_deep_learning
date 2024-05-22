import torch
import numpy as np

from typing import Iterable

class Classifier(torch.nn.Module):
    def __init__(self, *sizes:Iterable[int], activation=torch.nn.ReLU):
        super().__init__()
        layers = []

        for in_size, out_size in zip(sizes[:-2], sizes[1:-1]):
            layers.append(torch.nn.Linear(in_size, out_size, dtype=float))
            layers.append(activation())

        layers.append(torch.nn.Linear(sizes[-2], sizes[-1], dtype=float))
        layers.append(torch.nn.Softmax(dim=-1))

        self.model = torch.nn.Sequential(*layers)

    def fit(self, X_train, y_train, config, random_state:int=42):
        n_samples, n_features = X_train.shape

        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=config["lr"])

        # training
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        history = []
        for i in range(config["epochs"]):
            indices = np.random.permutation(n_samples)
            for j in range(0, n_samples, config["batch_size"]):
                X_batch = torch.tensor(X_train[indices[j:j+config["batch_size"]]])
                y_batch = torch.tensor(y_train[indices[j:j+config["batch_size"]]])

                optimizer.zero_grad()

                y_hat = self.model(X_batch)

                loss = loss_function(y_hat, y_batch)
                loss.backward()

                optimizer.step()

                print("Epoch", i+1, loss.detach().numpy())
            history.append(loss.detach().numpy())
        history = np.array(history)
        return history
    
    def score(self, X_test, y_test):
        with torch.no_grad():
            y_hat = self.model(torch.tensor(X_test))

        y_hat = np.argmax(y_hat.detach().cpu().numpy(), axis=1)

        return np.mean(y_hat == y_test)