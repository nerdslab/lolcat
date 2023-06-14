import torch
import torch.nn as nn

from torch_scatter import scatter_add
from torch_geometric.utils import softmax


####################
# Simple MLP model #
####################
class MLP(nn.Module):
    r"""Multi-layer perceptron model, with optional batchnorm layers.

    Args:
        hidden_layers (list): List of layer dimensions, from input layer to output layer. If first input size is -1,
            will use a lazy layer.
        bias (boolean, optional): If set to :obj:`True`, bias will be used in linear layers. (default: :obj:`True`).
        activation (torch.nn.Module, optional): Activation function. (default: :obj:`nn.ReLU`).
        batchnorm (boolean, optional): If set to :obj:`True`, batchnorm layers are added after each linear layer, before
            the activation (default: :obj:`False`).
        drop_last_nonlin (boolean, optional): If set to :obj:`True`, the last layer won't have activations or
            batchnorm layers. (default: :obj:`True`)

    Examples:
        >>> m = MLP([-1, 16, 64])
        MLP(
          (layers): Sequential(
            (0): LazyLinear(in_features=0, out_features=16, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=16, out_features=64, bias=True)
          )
        )
    """
    def __init__(self, hidden_layers, *, bias=True, activation=nn.ReLU(True), batchnorm=False, drop_last_nonlin=True, dropout=0.):
        super().__init__()

        # build the layers
        layers = []
        for in_dim, out_dim in zip(hidden_layers[:-1], hidden_layers[1:]):
            if in_dim == -1:
                layers.append(nn.LazyLinear(out_dim, bias=bias))
            else:
                layers.append(nn.Linear(in_dim, out_dim, bias=bias))
            if batchnorm:
                layers.append(nn.BatchNorm1d(num_features=out_dim))
            if activation is not None:
                layers.append(activation)
            if dropout > 0.:
                layers.append(nn.Dropout(dropout))

        # remove activation and/or batchnorm layers from the last block
        if drop_last_nonlin:
            remove_layers = -(int(activation is not None) + int(batchnorm) + int(dropout>0.))
            if remove_layers:
                layers = layers[:remove_layers]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()


#####################
# Attention Pooling #
#####################
class MultiHeadGlobalAttention(torch.nn.Module):
    """Multi-Head Global pooling layer."""
    def __init__(self, in_channels, out_channels, heads=1):
        super(MultiHeadGlobalAttention, self).__init__()
        self.heads = heads

        self.gate_nn = nn.Sequential(
            nn.Linear(in_channels, heads * in_channels, bias=False),
            nn.PReLU(),
            nn.Linear(heads * in_channels, heads, bias=False),
        )

        self.nn = MLP([in_channels, out_channels * heads, out_channels * heads])


    def forward(self, x, batch, return_attention=False):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = batch[-1].item() + 1

        gate = self.gate_nn(x)
        x = self.nn(x).view(x.size(0), self.heads, -1)

        score = softmax(gate, batch, num_nodes=size, dim=0)

        score = score.unsqueeze(-1)
        out = scatter_add(score * x, batch, dim=0, dim_size=size)

        out = out.view(size, -1)

        if not return_attention:
            return out
        else:
            return out, gate, score

    def __repr__(self):
        return '{}(gate_nn={}, nn={})'.format(self.__class__.__name__,
                                              self.gate_nn, self.nn)


################
# LOLCAT model #
################
class LOLCAT(nn.Module):
    """LOLCAT model. It consists of an encoder, a pooling layer and a classifier. The pooling layer is a multi-head
    global attention layer, which computes a global embedding for each cell. Both the encoder and the classifier 
    are MLPs."""
    def __init__(self, encoder, classifier, pool):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.pool = pool

    def forward(self, x, batch, return_attention=False):
        """Forward pass. If return_attention is True, the attention scores and gate values are returned as well.
        
        Args:
            x (torch.Tensor): Input tensor of shape (num_nodes, num_features).
            batch (torch.Tensor): Batch tensor of shape (num_nodes,).
            return_attention (bool, optional): If set to :obj:`True`, the attention scores and gate values are returned.
                (default: :obj:`False`).
        """
        emb = self.encoder(x)  # all trial sequences are encoded

        # compute global cell-wise embedding
        if return_attention:
            global_emb, gate, score = self.pool(emb, batch, return_attention=True)
        else:
            global_emb = self.pool(emb, batch)

        # classify
        logits = self.classifier(global_emb)

        if return_attention:
            return logits, {'global_emb': global_emb, 'attention': score, 'gate': gate}
        else:
            return logits, {'global_emb': global_emb}
