import numpy as np
import torch

from sklearn.cluster import KMeans

from shapelet_extraction import get_shapelets_mts

class PNTripletLoss(torch.nn.modules.loss._Loss):

    def __init__(self):
        super(PNTripletLoss, self).__init__()

    # Encode x
    def _encode(self, x:np.ndarray, encoder:torch.nn.Module):
        if len(x.shape) == 1:
            return encoder(x.reshape(1,1,x.shape[0]))
        elif len(x.shape) == 2:
            return encoder(x.reshape(x.shape[0],1,x.shape[1]))
        else:
            return encoder(x)
        
    # Get the number of samples to take (maximum 50)
    def _get_nr_of_samples(self, all_samples:int):
        return np.min([50,int(all_samples/5+1)])
    
    # Get the maximum pairwise normalized distance between the given shapelets
    def _get_pairwise_max_distance(self, shapelets:np.ndarray):
        max_distance = -1
        for i in range(len(shapelets)-1):
            for j in range(i+1,len(shapelets)):
                distance = torch.linalg.vector_norm(shapelets[i] - shapelets[j])
                if distance > max_distance:
                    max_distance = distance
        return max_distance

    def forward(self, batch:np.ndarray, encoder:torch.nn.Module):
        """
        Forward pass of the loss function, encodes the shapelets in the batch
        and calculates the loss based on the encoding. Good encodings result
        in similar shapelets being closer together in latent space.
        See the original paper for details on the loss function:
        ShapeNet: A Shapelet-Neural Network Approach for Multivariate Time Series Classification

        `batch` is an ndarray with the shape (`N`, `C`, `T`), where `N` is the
        batch size, `C` is the number of input channels, and `T` is the number
        of timesteps.

        `encoder` is a pytorch Module (CausalCNNEncoder).

        Outputs a double-valued loss with the gradient attached.
        """
        # TODO: Maybe use multiple shapelet lengths?
        shapelet_lengths = [5]

        # Enforced margin
        mu = 0.2
        # Regularizer between positive and negative intra-cluster distance
        lambda_ = 1

        # Total loss
        loss = torch.DoubleTensor([0])

        for shapelet_length in shapelet_lengths:
            # Create shapelets from the multivariate data
            shapelets = get_shapelets_mts(batch, l=shapelet_length)

            # Cluster the shapelets with kmeans (always 2 clusters, positive and negative)
            n_clusters = 2
            kmeans = KMeans(n_clusters, n_init="auto")
            kmeans.fit(shapelets)
            kmeans_labels = kmeans.labels_
            label,count = np.unique(kmeans_labels, return_counts=True)
            # Count number of shapelets by cluster
            shapelets_by_cluster = dict(zip(label,count))

            shapelets = torch.tensor(shapelets)

            # Loss by cluster
            loss_cluster = torch.DoubleTensor([0])
            # For each of the two clusters (could also do more theoretically)
            for i in range(n_clusters):
                if shapelets_by_cluster[i] < 2:
                    continue
                cluster_i = shapelets[kmeans_labels == i]
                distance_i = kmeans.transform(cluster_i)[:,i]


                # --- POSITIVE CLUSTER ---
                k_positive = self._get_nr_of_samples(shapelets_by_cluster[i])

                # get the k closest positive shapelets to the cluster center
                positive_indices = np.argpartition(distance_i, k_positive)[:(k_positive+1)]

                # encode the closest positive shapelet (anchor)
                # and the other positive shapelets (representatives)
                positive_shapelets_enc = self._encode(shapelets[positive_indices], encoder)
                x = positive_shapelets_enc[0]
                positive_representatives = positive_shapelets_enc[1:]

                # Calculate d_ap (mean normalized distance between anchor and positive representatives)
                d_ap = torch.sum(torch.linalg.vector_norm(x - positive_representatives, dim=1))
                d_ap = d_ap / k_positive

                # Intra cluster distance of positive cluster
                # d_pos is the normalized distance between the two most distant encoded positive shapelets
                d_pos = self._get_pairwise_max_distance(positive_representatives)
                
                # --- NEGATIVE CLUSTERS ---
                d_neg = torch.DoubleTensor([0])
                d_an = torch.DoubleTensor([0])
                for k in range(n_clusters):
                    # Same cluster, continue
                    if k == i:
                        continue
                    k_negative = self._get_nr_of_samples(shapelets_by_cluster[k])

                    # Randomly select a sample of k_negative shapelets
                    negative_shapelets = shapelets[kmeans_labels == k]
                    negative_indices = np.random.choice(len(negative_shapelets),k_negative,replace=False)
                    negative_representatives = self._encode(negative_shapelets[negative_indices], encoder)

                    # Calculate d_an (mean normalized distance between anchor and negative representatives)
                    dist_cluster_k_negative = torch.sum(torch.linalg.vector_norm(x - negative_representatives, dim=1))
                    d_an += dist_cluster_k_negative / k_negative

                    # Intra cluster distance of negative cluster
                    # d_neg is the normalized distance between the two most distant encoded negative shapelets
                    d_neg += self._get_pairwise_max_distance(negative_representatives)

                d_an = d_an / (n_clusters-1)
                loss_cluster += torch.log((d_ap + mu) / d_an)
                loss_cluster += lambda_ * (d_pos + d_neg)

            loss += loss_cluster
        return loss