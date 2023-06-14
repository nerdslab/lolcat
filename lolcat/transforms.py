import copy

import torch


class Compose:
    r"""Composes several transforms together. The transforms are applied in the order they are given."""
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


#################
# Normalization #
#################
class Normalize:
    """Normalizes the data by subtracting the mean and dividing by the standard deviation. If inplace is False, a copy 
    of the data is returned.
        
    Args:
        mean (torch.Tensor): The mean of the data.
        std (torch.Tensor): The standard deviation of the data.
        inplace (bool, optional): If True, the data is modified in place. (default: False)
    """
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, data):
        if not self.inplace:
            data = copy.deepcopy(data)
        x = data.x
        x = (x-self.mean.to(x.device)) / self.std.to(x.device)
        # If the standard deviation is zero, the data is set to zero. 
        x[:, self.std.squeeze() == 0.] = 0.
        data.x = x
        return data

    def unnormalize_x(self, x):
        return (x * self.std.to(x.device)) + self.mean.to(x.device)
    

def compute_mean_std(dataset, indices=None):
    r"""Computes the mean and standard deviation of the dataset. If indices is not None, only the data at the given
    indices is used to compute the mean and standard deviation.
        
    Args:
        dataset (torch_geometric.data.Dataset): The dataset.
        indices (list, optional): The indices of the data to use. If None, all data is used. (default: None)
    
    Returns:
        mean (torch.Tensor): The mean of the data.
        std (torch.Tensor): The standard deviation of the data.
    """
    if indices is None:
        indices = torch.arange(len(dataset))
    x_list = []
    for idx in indices:
        x = dataset[idx].x
        x_list.append(x)
    x = torch.cat(x_list, dim=0)
    std, mean = torch.std_mean(x, dim=0, unbiased=False, keepdim=True)
    return mean, std


###########
# Dropout #
###########
class Dropout:
    r"""Applies dropout to the data by removing, with probability ``dropout_p``, a fraction of the observed trials. 

    Args:
        dropout_p (float): The probability of removing a trial.
        apply_p (float, optional): The probability of applying the dropout. (default: 1.0)
        randomized (bool, optional): If True, the dropout probability is multiplied by a random number between 0 and 1.
            (default: False)
        inplace (bool, optional): If True, the data is modified in place. (default: False)
    """
    def __init__(self, dropout_p, apply_p=1.0, randomized=False, inplace=False):
        self.dropout_p = dropout_p
        self.apply_p = apply_p
        self.randomized = randomized
        self.inplace = inplace

    def __call__(self, data):
        if not self.inplace:
            data = copy.deepcopy(data)

        x = data.x
        if self.apply_p == 1. or torch.rand(1) < self.apply_p:
            dropout_p = self.dropout_p
            if self.randomized:
                dropout_p *= torch.rand(1)

            # dropout entire trials
            dropout_mask = torch.empty((x.size(0),), dtype=torch.float32, device=x.device).uniform_(0, 1) > dropout_p
            data.x = x[dropout_mask]
        return data
