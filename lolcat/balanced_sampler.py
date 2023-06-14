from typing import Iterator, Sequence

import torch
from torch.utils.data.sampler import Sampler


class DynamicBalancedSampler(Sampler[int]):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """
    indices: Sequence[int]

    def __init__(self, indices, group, init_weights=None, num_samples=None) -> None:
        super().__init__(None)

        self.indices = indices
        self.group = group

        # get inverse map for labels
        num_groups = torch.max(self.group) + 1
        self.group_sets = []
        for i in range(num_groups):
            self.group_sets.append(indices[torch.where(self.group == i)[0]])

        self.group_sizes = torch.tensor([group_set.size(0) for group_set in self.group_sets])
        self.group_ratio = self.group_sizes / self.group_sizes.sum()

        # init factors
        self.weights = 5 * torch.ones_like(self.group_ratio)

        # init_weights if init_weights is not None else self.init_oversampling_factor()

        if init_weights is not None:
            self.weights = init_weights

        self.indices = self.resample(self.weights)
        self.num_samples = num_samples

        self._step = 0

    def resample(self, weights):
        indices = []
        for i in range(len(self.group_sets)):
            group_set = self.group_sets[i]
            num_samples_in_group = group_set.size(0)

            weight = weights[i]
            weight_n, weight_f = int(weight), weight % 1.

            indices.append(torch.repeat_interleave(group_set, weight_n))
            indices.append(group_set[torch.randperm(num_samples_in_group)[:int(num_samples_in_group*weight_f)]])
        return torch.cat(indices)

    def __iter__(self) -> Iterator[int]:
        return (self.indices[i] for i in torch.randperm(len(self.indices))[:len(self)])

    def __len__(self) -> int:
        return self.num_samples if self.num_samples is not None else len(self.indices)

    def step(self, scores=None):
        self._step += 1

        if scores is not None:
            weights = 1 - scores
            weights = weights / weights.sum()
            weights = weights - weights.mean()
            self.weights = (self.weights + weights)

        self.indices = self.resample(self.weights)
