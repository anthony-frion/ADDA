from abc import ABC, abstractmethod
from textwrap import indent
from typing import List

import torch
from adda.system.state import State
from adda.util.data import as_TensorDict, impose_batch_size
from adda.util.state_space import same_shape, weighted_sse
from tensordict import TensorDict
from torch import randn_like, Tensor


class Distribution(ABC):
    """Base class for distributions over a TensorDict."""

    @abstractmethod
    def log_prob(self, x: TensorDict) -> float:
        """Compute log probability of state, summing over all non-batch axes.

        Args:
            x (State): input state
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self) -> TensorDict:
        """Sample from distribution on states."""
        raise NotImplementedError

    @property
    @abstractmethod
    def mean(self):
        """Mean of distribution on states.

        Requires a time_axis attribute.
        """
        raise NotImplementedError

    def __getitem__(self, key):
        """Indexing to obtain marginal distribution over a subset of variables."""
        raise NotImplementedError


class Gaussian(Distribution):
    """Normal distribution.

    Mu/Sigma are tensordicts, with optional batch dimensions.

    Args:
        mu (TensorDict): mean of distribution for each system variable
        sigma (TensorDict): covariance matrix for each pair of variables
    """

    def __init__(self, mu: TensorDict, sigma: TensorDict):
        """Initialize a Gaussian object. This function is not implemented since Gaussian is an abstract class, but it is
        defined for the classes inheriting from Gaussian.

        Args:
            mu (Tensordict): mean of distribution for each system variable
            sigma (TensorDict): covariance matrix for each pair of variables.
                Depending on the covariance structure, this could be represented more simply,
                e.g. as sets of diagonal coefficient for diagonal covariance structures.
        """
        raise NotImplementedError

    def log_prob(self, x: TensorDict, normalized: bool = False) -> Tensor:
        """Compute log probability, summing over all non-batch axes.

        Args:
            x (TensorDict): variables for which to evaluate the density
            normalized (bool): whether to normalize log probability
        """
        raise NotImplementedError

    def sample(self, n_samples: int = None) -> State:
        """Sample from the distribution on states.

        Args:
            n_samples (int): number of samples to draw from the distribution
        """
        raise NotImplementedError

    @property
    def mean(self):
        """Mean of distribution."""
        return self.mu

    def __repr__(self) -> str:
        """Defines out to print a Distribution object."""
        raise NotImplementedError

    def __getitem__(self, key):
        raise NotImplementedError


class DiagonalGaussian(Gaussian):
    """Diagonal normal distribution.

    Mu/sigma are tensordicts, with optional batch dimensions.

    Args:
        mu (TensorDict): mean of distribution per each system variable
        sigma (TensorDict): standard deviation of distribution per each system variable
    """

    def __init__(self, mu: TensorDict, sigma: TensorDict):
        """Initialize a DiagonalGaussian.

        Args:
            mu (Tensordict): mean of distribution for each system variable
            sigma (TensorDict): diagonal covariance coefficients
        """
        self.mu, self.sigma = as_TensorDict(mu), as_TensorDict(sigma)
        assert same_shape(self.mu, self.sigma), "mu/sigma mismatch"

    def log_prob(self, x: TensorDict, normalized: bool = False) -> Tensor:
        """Compute log probability, summing over all non-batch axes.

        Args:
            x (TensorDict): variables for which to evaluate the density
            normalized (bool): whether to normalize log probability
        """
        if normalized:
            raise NotImplementedError
        return -weighted_sse(x, self.mu, self.sigma)

    def sample(self, n_samples: int = None) -> State:
        """Sample from the distribution on states.

        Args:
            n_samples (int): number of samples to draw from the distribution
        """
        mu, sigma = self.mu, self.sigma
        if n_samples is not None:
            assert mu.shape[0] == 1, "batch dimension of mu/sigma must have length 1 when n_samples is specified"
            new_shape = (n_samples, *mu.shape[1:])
            mu, sigma = mu.expand(new_shape), sigma.expand(new_shape)
        return mu + randn_like(mu) * sigma

    def __getitem__(self, key):
        """Indexing operator.

        Args:
            key: indexing key, which be applied to index the mean and s.d. of this distribution

        Returns:
            DiagonalGaussian: indexed distribution
        """
        return DiagonalGaussian(self.mu[key], self.sigma[key])

    def __repr__(self) -> str:
        """Defines how to print a DiagonalDistribution object."""
        string_mu = indent(f"mu: {self.mu}", 4 * " ")
        string_sigma = indent(f"sigma: {self.sigma}", 4 * " ")
        return f"{type(self).__name__}(\n{string_mu}\n{string_sigma})"


def impose_batch_sizes(distrib: Distribution, size: torch.Size, attrs: List) -> Distribution:
    """Assign a batch size to TensorDict attributes of an input Distribution object. Throw an error if one TensorDict
    has more dimensions than the desired batch_size.

    Args:
        distrib (Distribution): input
        size (torch.Size): desired batch_size
        attrs (List): list of names for the attributes on which to impose the batch sizes.

    Returns:
        Distribution: updated input
    """
    assert isinstance(distrib, Distribution), "input must be of type da_tools.probability.Distribution"
    for attr in attrs:
        if hasattr(distrib, attr):
            x = getattr(distrib, attr)
            setattr(distrib, attr, impose_batch_size(x, size))
    return distrib
