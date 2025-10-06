import torch
from scvi.distributions import NegativeBinomial as NegativeBinomialSCVI
from torch.distributions import Bernoulli, Distribution


def log_nb_positive(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    eps: float = 1e-8,
    log_fn: callable = torch.log,
    lgamma_fn: callable = torch.lgamma,
) -> torch.Tensor:
    """Log likelihood (scalar) of a minibatch according to a nb model.

    Parameters
    ----------
    x
        data
    mu
        mean of the negative binomial (has to be positive support) (shape: minibatch x vars)
    theta
        inverse dispersion parameter (has to be positive support) (shape: minibatch x vars)
    eps
        numerical stability constant
    log_fn
        log function
    lgamma_fn
        log gamma function
    """
    log = log_fn
    lgamma = lgamma_fn
    log_theta_mu_eps = log(theta + mu + eps)
    res = (
        theta * (log(theta + eps) - log_theta_mu_eps)
        + x * (log(mu + eps) - log_theta_mu_eps)
        + lgamma(x + theta)
        - lgamma(theta)
        - lgamma(x + 1)
    )

    return res


def log_min_exp(a, b, epsilon=1e-8):
    """Source: https://github.com/jornpeters/integer_discrete_flows"""
    y = a + torch.log(1 - torch.exp(b - a) + epsilon)
    return y


class MaskedZeroTruncatedNegativeBinomialSCVI(NegativeBinomialSCVI):
    """Negative Binomial distribution with optional zero-truncation.

    Parameters
    ----------
    mu
        mean of the negative binomial (has to be positive support)
    theta
        inverse dispersion parameter (has to be positive support)
    validate_args
        whether to validate input.
    """

    def __init__(
        self,
        mu: torch.Tensor,
        theta: torch.Tensor,
        logits: torch.Tensor | None = None,
        validate_args: bool | None = None,
    ):
        super().__init__(mu=mu, theta=theta, validate_args=validate_args)
        self.logits = logits

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            self._validate_sample(value)

        value_binary = (value > 0).to(value.dtype)
        bernoulli_dist = Bernoulli(logits=self.logits)
        log_prob_bernoulli = bernoulli_dist.log_prob(value_binary)

        log_prob = log_nb_positive(value, mu=self.mu, theta=self.theta, eps=self._eps)
        log_norm = torch.log1p(-((self.theta / (self.theta + self.mu + self._eps)) ** self.theta))
        log_prob_zero_truncated_nb = log_prob - log_norm
        return log_prob_bernoulli + (log_prob_zero_truncated_nb * value_binary)

    @torch.inference_mode()
    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """Sample from the distribution."""
        gamma_d = self._gamma()
        p_means = gamma_d.sample(sample_shape)

        # Clamping as distributions objects can have buggy behaviors when
        # their parameters are too high
        p_means = torch.nn.functional.softplus(p_means + self._eps)
        rate = torch.clamp(p_means, max=1e8)

        u = torch.distributions.Uniform(torch.exp(-rate), 1).sample(sample_shape)
        # Transform uniform samples
        t = -torch.log(u)
        # Sample from shifted Poisson
        samples = 1 + torch.poisson(rate - t + self._eps)

        mask = Bernoulli(logits=self.logits).sample(sample_shape)
        return samples * mask


class TruncatedDiscretizedLogistic(Distribution):
    arg_constraints = {
        "loc": torch.distributions.constraints.real,
        "scale": torch.distributions.constraints.positive,
    }  # No constraints for simplicity, but could be added

    def __init__(self, loc, scale, bin_size: float = 1.0, validate_args: bool | None = None):
        self.loc = loc
        self.scale = scale
        self.bin_size = bin_size
        self._eps = 1e-10  # Small constant for numerical stability

        batch_shape = torch.broadcast_shapes(loc.shape, scale.shape)
        super().__init__(batch_shape, validate_args=validate_args)

    def log_cdf(self, x):
        """Cumulative distribution function."""
        v = (x - self.loc) / self.scale
        return torch.nn.functional.logsigmoid(v)

    def log_prob(self, value):
        """

        Compute the pmf of a truncated discretized logistic distribution on the nonnegative integers.

        The pmf is given by (assuming the bin size equals 1):
            P(X=k) = (F(k + 0.5) - F(k - 0.5)) / (1 - F(-0.5))
        where
            F(x) = 1 / (1 + exp(-(x - mu)/scale))

        The logarithm of the pmf:
            log P(X=k) = log(F(k + 0.5) - F(k - 0.5)) - log(1 - F(-0.5))

        """
        # F(k + bin_size/2)
        value_plus = value + self.bin_size / 2.0
        cdf_plus = self.log_cdf(value_plus)
        # F(k - bin_size/2)
        value_minus = value - self.bin_size / 2.0
        cdf_minus = self.log_cdf(value_minus)
        # the nominator: (F(k + 0.5) - F(k - 0.5))
        log_nominator = log_min_exp(cdf_plus, cdf_minus)
        cdf_neg_half = self.log_cdf(-(self.bin_size / 2.0))
        # the denominator: (1 - F(-0.5))
        zeros = value * 0.0
        log_denominator = log_min_exp(zeros, cdf_neg_half)
        return log_nominator - log_denominator

    def sample(self, sample_shape=torch.Size()):
        """

        Generates samples using inverse transform sampling.

        Sampling steps:
            1. Draw u ~ Uniform(0, 1) with the appropriate shape.
            2. Compute t = u*(1 - F(-0.5)) + F(-0.5), where F(-0.5) is the logistic CDF at -0.5.
            3. Invert the logistic CDF: inv = loc + scale * log(t / (1 - t)).
            4. Return k = ceil(inv - 0.5) as the discretized sample.

        Note: The ceiling operation is non-differentiable, which is standard for sampling.

        """
        neg_half = torch.exp(self.log_cdf(-(self.bin_size / 2.0)))
        if torch.isnan(neg_half).any():
            print(f"NaN values in neg_half: {neg_half[torch.isnan(neg_half)]}")
        u = torch.distributions.Uniform(neg_half, torch.ones_like(neg_half)).sample(sample_shape)
        if torch.isnan(u).any():
            print(f"NaN values in u: {u[torch.isnan(u)]}")
        t = torch.clamp(u * (1 - neg_half) + neg_half, min=self._eps, max=1 - self._eps)
        if torch.isnan(t).any():
            print(f"NaN values in t: {t[torch.isnan(t)]}")
        # Compute inverse logistic
        inv_logistic = self.loc + self.scale * (torch.log(t) - torch.log(1 - t))
        if torch.isnan(inv_logistic).any():
            print(f"NaN values in inv_logistic: {inv_logistic[torch.isnan(inv_logistic)]}")
        x = torch.ceil(inv_logistic - (self.bin_size / 2.0))
        if torch.isnan(x).any():
            print(f"NaN values in x: {x[torch.isnan(x)]}")
            exit()
        return x


class MaskedZeroTruncatedDiscretizedLogistic(TruncatedDiscretizedLogistic):
    def __init__(self, loc, scale, logits, bin_size: float = 1.0, validate_args: bool | None = None):
        super().__init__(loc=loc, scale=scale, bin_size=bin_size, validate_args=validate_args)
        self.logits = logits

    def log_prob(self, value):
        mask = (value > 0).to(value.dtype)
        bernoulli_dist = Bernoulli(logits=self.logits)
        log_prob_bernoulli = bernoulli_dist.log_prob(mask)

        log_prob = super().log_prob(value)

        zeros = value * 0.0
        log_norm = log_min_exp(zeros, self.cdf(+(self.bin_size / 2.0)))

        log_prob_zero_truncated = log_prob - log_norm
        return log_prob_bernoulli + (log_prob_zero_truncated * mask)

    @torch.inference_mode()
    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        p_less_than_or_equal_zero = self.cdf(0.0 + self.bin_size / 2.0)

        # Generate uniform samples in the range [p_less_than_or_equal_zero, 1]
        # This effectively shifts the CDF to sample only from X>0
        u = p_less_than_or_equal_zero + (1.0 - p_less_than_or_equal_zero) * torch.rand(shape, device=self.loc.device)

        # Apply inverse CDF
        x = self.loc + self.scale * torch.log(u / (1 - u + self._eps) + self._eps)
        x = torch.floor(x / self.bin_size) * self.bin_size

        mask = Bernoulli(logits=self.logits).sample(sample_shape)
        return x * mask


class MixtureTruncatedDiscretizedLogistic(Distribution):
    def __init__(self, locs, scales, weights, bin_size: float = 1.0, validate_args: bool | None = None):
        # Check that we have same number of locs, scales, and weights
        if locs.shape[0] != scales.shape[0] or locs.shape[0] != weights.shape[0]:
            raise ValueError("Number of location, scale, and weight parameters must match")

        self.locs = locs
        self.scales = scales
        self.weights = weights
        self.bin_size = bin_size
        self.n_components = locs.shape[0]
        self._eps = 1e-8  # Small constant for numerical stability

        # Create component distributions
        self.component_distributions = [
            TruncatedDiscretizedLogistic(locs[i], scales[i], bin_size) for i in range(self.n_components)
        ]

        # Determine batch shape by broadcasting all parameters
        batch_shape = locs.shape[1:] if len(locs.shape) > 1 else torch.Size()
        super().__init__(batch_shape, validate_args=validate_args)

    def log_prob(self, value):
        # Calculate log_prob for each component
        component_log_probs = torch.stack(
            [component.log_prob(value) for component in self.component_distributions], dim=0
        )

        # Log mixture probability using log-sum-exp trick for numerical stability
        log_weights = torch.log_softmax(self.weights, dim=0)
        log_probs = log_weights.unsqueeze(-1) + component_log_probs
        return torch.logsumexp(log_probs, dim=0)

    @torch.inference_mode()
    def sample(self, sample_shape=torch.Size()):
        # Extend shape for sampling
        shape = self._extended_shape(sample_shape)

        # Sample component indices based on weights
        component_indices = torch.multinomial(
            torch.softmax(self.weights, dim=0), num_samples=torch.prod(torch.tensor(shape)).item(), replacement=True
        ).reshape(shape)

        # Initialize output tensor
        samples = torch.zeros(shape, device=self.locs.device)

        # Sample from each chosen component
        for i in range(self.n_components):
            mask = component_indices == i
            if mask.any():
                component_sample_shape = torch.sum(mask).item()
                component_samples = self.component_distributions[i].sample(torch.Size([component_sample_shape]))
                samples[mask] = component_samples

        return samples


class MixtureMaskedZeroTruncatedDiscretizedLogistic(Distribution):
    def __init__(self, locs, scales, weights, logits, bin_size: float = 1.0, validate_args: bool | None = None):
        # Check that we have same number of locs, scales, and weights
        if locs.shape[0] != scales.shape[0] or locs.shape[0] != weights.shape[0]:
            raise ValueError("Number of location, scale, and weight parameters must match")

        self.locs = locs
        self.scales = scales
        self.weights = weights
        self.logits = logits
        self.bin_size = bin_size
        self.n_components = locs.shape[0]
        self._eps = 1e-8  # Small constant for numerical stability

        self.component_distributions = [
            MaskedZeroTruncatedDiscretizedLogistic(locs[i], scales[i], logits, bin_size)
            for i in range(self.n_components)
        ]

        batch_shape = locs.shape[1:] if len(locs.shape) > 1 else torch.Size()
        super().__init__(batch_shape, validate_args=validate_args)

    def log_prob(self, value):
        # Calculate log_prob for each component
        component_log_probs = torch.stack(
            [component.log_prob(value) for component in self.component_distributions], dim=0
        )

        # Log mixture probability using log-sum-exp trick for numerical stability
        log_weights = torch.log_softmax(self.weights, dim=0)
        log_probs = log_weights.unsqueeze(-1) + component_log_probs
        return torch.logsumexp(log_probs, dim=0)

    @torch.inference_mode()
    def sample(self, sample_shape=torch.Size()):
        # Extend shape for sampling
        shape = self._extended_shape(sample_shape)

        component_indices = torch.multinomial(
            torch.softmax(self.weights, dim=0), num_samples=torch.prod(torch.tensor(shape)).item(), replacement=True
        ).reshape(shape)

        samples = torch.zeros(shape, device=self.locs.device)

        for i in range(self.n_components):
            mask = component_indices == i
            if mask.any():
                component_sample_shape = torch.sum(mask).item()
                component_samples = self.component_distributions[i].sample(torch.Size([component_sample_shape]))
                samples[mask] = component_samples

        return samples


class TruncatedDiscretizedNormal(Distribution):
    r"""

    A Discretized Gaussian distribution truncated to non-negative integers:

        P(X = k) =
          [ F((k+0.5 - mean)/std) - F((k-0.5 - mean)/std) ]
          / [ 1 - F((-0.5 - mean)/std) ],
        for k = 0, 1, 2, ...
    and 0 otherwise.

    where F is the standard Normal CDF parameterized by (loc=mean, scale=std).

    This effectively 'cuts off' any mass for k < 0.

    """

    arg_constraints = {"loc": torch.distributions.constraints.real, "scale": torch.distributions.constraints.positive}
    # The distribution is supported on nonnegative integers:
    support = torch.distributions.constraints.nonnegative_integer
    has_rsample = False

    def __init__(self, loc, scale, validate_args=None):
        self.loc, self.scale = torch.distributions.utils.broadcast_all(loc, scale)

        if validate_args:
            self._validate_args()

        super().__init__(batch_shape=self.loc.size(), validate_args=validate_args)

        # Create an underlying continuous Normal for CDF calls
        self._continuous_dist = torch.distributions.Normal(self.loc, self.scale)

        # Precompute the normalizing constant, i.e.,
        #   Z = 1 - F( ( -0.5 - mean ) / std )
        # for the truncated domain [0, infty)
        self._Z = 1.0 - self._continuous_dist.cdf(torch.tensor(-0.5, device=self.loc.device, dtype=self.loc.dtype))

    def pmf(self, value):
        """pmf(k) = P(X=k) for k >= 0, else 0."""
        # Ensure 'value' is a tensor in the same dtype:
        value = value.to(dtype=self.loc.dtype)

        # Mask out negative k:
        nonneg_mask = value >= 0

        # Prepare arrays for pmf results, set default = 0 for negative k
        pmf_val = torch.zeros_like(value)

        # For the valid positions, compute:
        # numerator = NormalCDF(k+0.5) - NormalCDF(k-0.5)
        # pmf(k) = numerator / Z
        if nonneg_mask.any():
            val_nonneg = value * nonneg_mask

            left_edge = val_nonneg - 0.5
            right_edge = val_nonneg + 0.5

            cdf_right = self._continuous_dist.cdf(right_edge)
            cdf_left = self._continuous_dist.cdf(left_edge)
            numer = cdf_right - cdf_left

            # We clamp to avoid negative or zero from numerical issues:
            numer = torch.clamp(numer, min=0.0)
            Z_val = torch.clamp(self._Z, min=1e-20)

            pmf_val_nonneg = numer / Z_val
            pmf_val = torch.where(nonneg_mask, pmf_val_nonneg, pmf_val)

        return pmf_val

    def log_prob(self, value):
        """Log pmf(k) = log( pmf(k) )."""
        pmf_val = self.pmf(value)
        # clamp to avoid log(0)
        return torch.log(torch.clamp(pmf_val, min=1e-20))

    def cdf(self, value):
        """CDF(k) = P(X <= k). For k < 0, this is 0; otherwise, it is [ NormalCDF(k + 0.5) - NormalCDF(-0.5) ] / Z."""
        value = value.to(dtype=self.loc.dtype)
        # For k < 0 => cdf=0
        cdf_val = torch.zeros_like(value)

        # For k >= 0:
        nonneg_mask = value >= 0
        if nonneg_mask.any():
            val_nonneg = value[nonneg_mask]
            cdf_top = self._continuous_dist.cdf(val_nonneg + 0.5)
            cdf_bottom = self._continuous_dist.cdf(torch.tensor(-0.5, device=value.device, dtype=value.dtype))
            numer = cdf_top - cdf_bottom
            numer = torch.clamp(numer, min=0.0)
            denom = torch.clamp(self._Z, min=1e-20)
            cdf_val[nonneg_mask] = numer / denom

        return cdf_val

    def sample(self, sample_shape=torch.Size()):
        """Draw samples using a truncated range method:

        - We'll sample from k=0 up to k = floor(mu + 5*sigma), or 0 if that is negative.
        - Compute pmf for all these k's, then sample from Categorical.
        """
        shape = self._extended_shape(sample_shape)

        cdf_neg_half = self._continuous_dist.cdf(torch.tensor(-0.5, device=self.loc.device, dtype=self.loc.dtype))
        u = torch.rand(shape, device=self.loc.device, dtype=self.loc.dtype)
        u_transformed = cdf_neg_half + u * (1.0 - cdf_neg_half)
        continuous_samples = self._continuous_dist.icdf(u_transformed)
        discrete_samples = torch.round(continuous_samples)
        discrete_samples = torch.clamp(discrete_samples, min=0)
        return discrete_samples.to(torch.int32)
