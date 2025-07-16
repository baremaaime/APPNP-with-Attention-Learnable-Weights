"""APPNP and PPNP layers."""

import math
import torch
from torch_sparse import spmm
# from utils import create_propagator_matrix

def uniform(size, tensor):
    """
    Uniform weight initialization.
    :param size: Size of the tensor.
    :param tensor: Tensor initialized.
    """
    stdv = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)

class DenseFullyConnected(torch.nn.Module):
    """
    Abstract class for PageRank and Approximate PageRank networks.
    :param in_channels: Number of input channels.
    :param out_channels: Number of output channels.
    :param density: Feature matrix structure.
    """
    def __init__(self, in_channels, out_channels):
        super(DenseFullyConnected, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.define_parameters()
        self.init_parameters()

    def define_parameters(self):
        """
        Defining the weight matrices.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(self.out_channels))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        uniform(self.out_channels, self.bias)

    def forward(self, features):
        """
        Doing a forward pass.
        :param features: Feature matrix.
        :return filtered_features: Convolved features.
        """
        filtered_features = torch.mm(features, self.weight_matrix)
        filtered_features = filtered_features + self.bias
        return filtered_features

class SparseFullyConnected(torch.nn.Module):
    """
    Abstract class for PageRank and Approximate PageRank networks.
    :param in_channels: Number of input channels.
    :param out_channels: Number of output channels.
    :param density: Feature matrix structure.
    """
    def __init__(self, in_channels, out_channels):
        super(SparseFullyConnected, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.define_parameters()
        self.init_parameters()

    def define_parameters(self):
        """
        Defining the weight matrices.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(self.out_channels))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        uniform(self.out_channels, self.bias)

    def forward(self, feature_indices, feature_values):
        """
        Making a forward pass.
        :param feature_indices: Non zero value indices.
        :param feature_values: Matrix values.
        :return filtered_features: Output features.
        """
        number_of_nodes = torch.max(feature_indices[0]).item()+1
        number_of_features = torch.max(feature_indices[1]).item()+1
        filtered_features = spmm(index = feature_indices,
                                 value = feature_values,
                                 m = number_of_nodes,
                                 n = number_of_features,
                                 matrix = self.weight_matrix)
        filtered_features = filtered_features + self.bias
        return filtered_features

class APPNPModel(torch.nn.Module):
    def __init__(self, args, number_of_labels, number_of_features, graph, device):
        super(APPNPModel, self).__init__()
        self.args = args
        self.number_of_labels = number_of_labels
        self.number_of_features = number_of_features
        self.graph = graph
        self.device = device
        self.node_count = graph.number_of_nodes()
        
        # Learnable attention matrix for node pairs (symmetric)
        self.attention_weights = torch.nn.Parameter(torch.ones(self.node_count, self.node_count))

        self.setup_layers()
        self.setup_propagator()

    def setup_layers(self):
        self.layer_1 = SparseFullyConnected(self.number_of_features, self.args.layers[0])
        self.layer_2 = DenseFullyConnected(self.args.layers[1], self.number_of_labels)

    def setup_propagator(self):
        self.propagator = self.create_propagator_matrix(
            self.graph, self.args.alpha, self.args.model
        )
        if self.args.model == "exact":
            self.propagator = self.propagator.to(self.device)
        else:
            self.edge_indices = self.propagator["indices"].to(self.device)
            self.edge_weights = self.propagator["values"].to(self.device)

    def create_adjacency_matrix(self, graph):
        index_1 = [edge[0] for edge in graph.edges()] + [edge[1] for edge in graph.edges()]
        index_2 = [edge[1] for edge in graph.edges()] + [edge[0] for edge in graph.edges()]
        values = [1] * len(index_1)
        node_count = max(max(index_1) + 1, max(index_2) + 1)
        return sparse.coo_matrix((values, (index_1, index_2)), shape=(node_count, node_count), dtype=np.float32)

    def normalize_adjacency_matrix(self, A, I):
        A_tilde = A + I
        degrees = np.array(A_tilde.sum(axis=1)).flatten()
        D = sparse.diags(np.power(degrees, -0.5))
        return D.dot(A_tilde).dot(D)

    def create_propagator_matrix(self, graph, alpha, model):
        A = self.create_adjacency_matrix(graph)
        I = sparse.eye(A.shape[0])
        A_tilde_hat = self.normalize_adjacency_matrix(A, I)

        # Element-wise multiplication with learnable attention (applied only to non-zero elements)
        B_dense = self.attention_weights.detach().cpu().numpy()
        B_sparse = sparse.coo_matrix(B_dense)  # Convert to sparse
        A_tilde_hat = A_tilde_hat.multiply(B_sparse)

        if model == "exact":
            P = (I - (1 - alpha) * A_tilde_hat).todense()
            P = alpha * torch.inverse(torch.FloatTensor(P))
            return P
        else:
            A_tilde_hat = sparse.coo_matrix(A_tilde_hat)
            indices = np.vstack((A_tilde_hat.row, A_tilde_hat.col))
            values = A_tilde_hat.data
            return {
                "indices": torch.LongTensor(indices),
                "values": torch.FloatTensor(values)
            }

    def forward(self, feature_indices, feature_values):
        feature_values = torch.nn.functional.dropout(
            feature_values, p=self.args.dropout, training=self.training
        )

        latent_features_1 = self.layer_1(feature_indices, feature_values)
        latent_features_1 = torch.nn.functional.relu(latent_features_1)
        latent_features_1 = torch.nn.functional.dropout(
            latent_features_1, p=self.args.dropout, training=self.training
        )
        latent_features_2 = self.layer_2(latent_features_1)

        if self.args.model == "exact":
            self.predictions = torch.mm(self.propagator, latent_features_2)
        else:
            localized_predictions = latent_features_2
            edge_weights = torch.nn.functional.dropout(
                self.edge_weights, p=self.args.dropout, training=self.training
            )
            for _ in range(self.args.iterations):
                new_features = spmm(
                    index=self.edge_indices,
                    value=edge_weights,
                    n=localized_predictions.size(0),
                    m=localized_predictions.size(0),
                    matrix=localized_predictions,
                )
                localized_predictions = (
                    (1 - self.args.alpha) * new_features
                    + self.args.alpha * latent_features_2
                )
            self.predictions = localized_predictions

        self.predictions = torch.nn.functional.log_softmax(self.predictions, dim=1)
        return self.predictions