from typing import Any, Optional, Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp


def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def ensemblize(cls, num_qs, in_axes=None, out_axes=0, **kwargs):
    """Ensemblize a module."""
    return nn.vmap(
        cls,
        variable_axes={'params': 0, 'intermediates': 0},
        split_rngs={'params': True},
        in_axes=in_axes,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class FourierFeatures(nn.Module):
    # used for timestep embedding
    output_size: int = 64
    learnable: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if self.learnable:
            w = self.param('kernel', nn.initializers.normal(0.2),
                           (self.output_size // 2, x.shape[-1]), jnp.float32)
            f = 2 * jnp.pi * x @ w.T
        else:
            half_dim = self.output_size // 2
            f = jnp.log(10000) / (half_dim - 1)
            f = jnp.exp(jnp.arange(half_dim) * -f)
            f = x * f
        return jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)



class Identity(nn.Module):
    """Identity layer."""

    def __call__(self, x):
        return x


class MLP(nn.Module):
    """Multi-layer perceptron.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activations: Activation function.
        activate_final: Whether to apply activation to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False
    use_zero_output: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            if i == len(self.hidden_dims) - 1 and self.use_zero_output:
                kernel_init = nn.initializers.zeros
            else:
                kernel_init = self.kernel_init
            x = nn.Dense(size, kernel_init=kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
            if i == len(self.hidden_dims) - 2:
                self.sow('intermediates', 'feature', x)
        return x


class LogParam(nn.Module):
    """Scalar parameter module with log scale."""

    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_value = self.param('log_value', init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        return jnp.exp(log_value)


class TransformedWithMode(distrax.Transformed):
    """Transformed distribution with mode calculation."""

    def mode(self):
        return self.bijector.forward(self.distribution.mode())


class Actor(nn.Module):
    """Gaussian actor network.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        log_std_min: Minimum value of log standard deviation.
        log_std_max: Maximum value of log standard deviation.
        tanh_squash: Whether to squash the action with tanh.
        state_dependent_std: Whether to use state-dependent standard deviation.
        const_std: Whether to use constant standard deviation.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    encoder: nn.Module = None

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        else:
            if not self.const_std:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(
        self,
        observations,
        temperature=1.0,
    ):
        """Return action distributions.

        Args:
            observations: Observations.
            temperature: Scaling factor for the standard deviation.
        """
        if self.encoder is not None:
            inputs = self.encoder(observations)
        else:
            inputs = observations
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))

        return distribution


class Value(nn.Module):
    """Value/critic network.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
        num_ensembles: Number of ensemble components.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    layer_norm: bool = True
    num_ensembles: int = 2
    encoder: nn.Module = None
    output_dim: int = 1
    use_zero_output: bool = False

    def setup(self):
        mlp_class = MLP
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)
        value_net = mlp_class(
            (*self.hidden_dims, self.output_dim), 
            activate_final=False, 
            layer_norm=self.layer_norm,
            use_zero_output=self.use_zero_output
        )

        self.value_net = value_net

    def __call__(self, observations, actions=None):
        """Return values or critic values.

        Args:
            observations: Observations.
            actions: Actions (optional).
        """
        if self.encoder is not None:
            inputs = [self.encoder(observations)]
        else:
            inputs = [observations]
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.value_net(inputs)
        
        if self.output_dim == 1:
             v = v.squeeze(-1)

        return v


class ActorVectorField(nn.Module):
    """Actor vector field network for flow matching.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    encoder: nn.Module = None
    use_fourier_features: bool = False
    fourier_feature_dim: int = 64

    def setup(self) -> None:
        self.mlp = MLP((*self.hidden_dims, self.action_dim), activate_final=False, layer_norm=self.layer_norm)
        if self.use_fourier_features:
            self.ff = FourierFeatures(self.fourier_feature_dim)

    @nn.compact
    def __call__(self, observations, actions, times=None, is_encoded=False):
        """Return the vectors at the given states, actions, and times (optional).

        Args:
            observations: Observations.
            actions: Actions.
            times: Times (optional).
            is_encoded: Whether the observations are already encoded.
        """
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)
        if times is None:
            inputs = jnp.concatenate([observations, actions], axis=-1)
        else:
            if self.use_fourier_features:
                times = self.ff(times)
            inputs = jnp.concatenate([observations, actions, times], axis=-1)

        v = self.mlp(inputs)

        return v


class BROValue(nn.Module):
    """Value/critic network.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
        num_ensembles: Number of ensemble components.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    depth: int = 2
    num_ensembles: int = 2
    encoder: nn.Module = None
    output_dim: int = 1
    use_zero_output: bool = False

    def setup(self):
        mlp_class = BRONet
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)
        value_net = mlp_class(
            hidden_dims=self.hidden_dims,
            depth=self.depth,
            output_dim=self.output_dim,
            use_zero_output=self.use_zero_output,
        )

        self.value_net = value_net

    def __call__(self, observations, actions=None):
        """Return values or critic values.

        Args:
            observations: Observations.
            actions: Actions (optional).
        """
        if self.encoder is not None:
            inputs = [self.encoder(observations)]
        else:
            inputs = [observations]
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.value_net(inputs)
        if self.output_dim == 1:
            v = v.squeeze(-1)
        return v
    
    

class QuantileValue(nn.Module):
    """Quantile Value/Critic network.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        num_atoms: Number of quantiles (atoms).
        layer_norm: Whether to apply layer normalization.
        num_ensembles: Number of ensemble components.
        encoder: Optional encoder module.
    """
    hidden_dims: Sequence[int]
    num_atoms: int = 20
    layer_norm: bool = True
    num_ensembles: int = 2
    encoder: nn.Module = None

    def setup(self):
        mlp_class = MLP
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)
        
        # Output num_atoms instead of 1
        self.value_net = mlp_class((*self.hidden_dims, self.num_atoms), activate_final=False, layer_norm=self.layer_norm)

    def __call__(self, observations, actions=None):
        if self.encoder is not None:
            inputs = [self.encoder(observations)]
        else:
            inputs = [observations]
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        # Returns (num_ensembles, batch, num_atoms)
        return self.value_net(inputs)


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block: causal self-attention + MLP, both residual.

    Attributes:
        hidden_dim: Token/residual-stream width.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden width as a multiple of hidden_dim.
        layer_norm: Whether to apply LayerNorm before each sub-block (pre-LN). If False, both
            sub-blocks still get residual connections but skip normalization -- not recommended,
            kept only for parity with this file's other modules' layer_norm toggle.
    """

    hidden_dim: int
    num_heads: int
    mlp_ratio: int = 4
    layer_norm: bool = True

    @nn.compact
    def __call__(self, x, mask):
        norm = nn.LayerNorm if self.layer_norm else Identity
        h = norm()(x)
        attn = nn.SelfAttention(num_heads=self.num_heads, qkv_features=self.hidden_dim, out_features=self.hidden_dim)(
            h, mask=mask
        )
        x = x + attn
        h = norm()(x)
        h = nn.Dense(self.hidden_dim * self.mlp_ratio)(h)
        h = nn.gelu(h)
        h = nn.Dense(self.hidden_dim)(h)
        x = x + h
        return x


class CausalChunkQTransformer(nn.Module):
    """Causal-Transformer chunked Q-network (ACSAC, arXiv:2605.11009, Sec 4.1).

    Tokenizes a state and an H-length action chunk as [state, a_1, ..., a_H] (H = max(chunk_sizes)),
    runs them through a causal Transformer (position h's representation depends only on
    state, a_1, ..., a_h -- never a_{h+1:}), and reads a scalar Q_h(s, a_{1:h}) off each requested
    head-position h via one *shared* linear read-out applied per-token-position (not a separate head
    per h -- the positions differ in what they causally see, not in which weights read them out).
    One forward pass over the full H-length chunk yields every requested head simultaneously.

    Attributes:
        chunk_sizes: Head positions to read out, e.g. (16, 8, 4, 2, 1). Need not be contiguous or
            sorted (sorting only matters to callers, not to this module) -- every h must satisfy
            1 <= h <= max(chunk_sizes).
        hidden_dim: Token width.
        num_layers: Number of TransformerBlocks.
        num_heads: Attention heads per block.
        mlp_ratio: Per-block MLP hidden width as a multiple of hidden_dim.
        layer_norm: Whether TransformerBlocks use pre-LN.
        encoder: Optional encoder module for the observation, applied before tokenization.
    """

    chunk_sizes: Sequence[int]
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 8
    mlp_ratio: int = 4
    layer_norm: bool = True
    encoder: nn.Module = None

    def setup(self):
        self.horizon = max(self.chunk_sizes)
        self.state_proj = nn.Dense(self.hidden_dim, kernel_init=default_init())
        self.action_proj = nn.Dense(self.hidden_dim, kernel_init=default_init())
        self.pos_embed = self.param('pos_embed', nn.initializers.normal(0.02), (self.horizon + 1, self.hidden_dim))
        self.blocks = [
            TransformerBlock(hidden_dim=self.hidden_dim, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, layer_norm=self.layer_norm)
            for _ in range(self.num_layers)
        ]
        self.final_norm = nn.LayerNorm() if self.layer_norm else Identity()
        self.q_head = nn.Dense(1, kernel_init=default_init(1e-2))

    def all_positions(self, observations, actions):
        """Compute Q at *every* position 0..horizon in one forward pass (position 0 is the state
        token and is meaningless -- kept only so index h directly means "Q_h" with no off-by-one).

        Args:
            observations: (..., obs_dim).
            actions: (..., horizon, action_dim) -- the full H-length action chunk. Only the first h
                actions causally influence position h's output; positions after h are
                architecturally invisible to it (see scripts/test_causal_chunk_q_transformer.py).

        Returns:
            Array of shape (..., horizon + 1). Exposed as a separate method (rather than folded into
            __call__) so callers needing a *traced* (e.g. randomly-sampled) head index can gather
            straight out of this array -- a Python dict, as __call__ returns, can only be keyed by a
            static Python int.
        """
        if self.encoder is not None:
            observations = self.encoder(observations)
        state_tok = self.state_proj(observations)[..., None, :]  # (..., 1, hidden_dim)
        action_tok = self.action_proj(actions)  # (..., horizon, hidden_dim)
        x = jnp.concatenate([state_tok, action_tok], axis=-2)  # (..., horizon + 1, hidden_dim)
        x = x + self.pos_embed  # broadcasts over leading batch dims

        mask = nn.make_causal_mask(jnp.ones(x.shape[:-1]))
        for block in self.blocks:
            x = block(x, mask=mask)
        x = self.final_norm(x)

        return self.q_head(x).squeeze(-1)  # (..., horizon + 1)

    def __call__(self, observations, actions):
        """Compute Q_h(s, a_{1:h}) for every h in chunk_sizes in one forward pass.

        Args:
            observations: (..., obs_dim).
            actions: (..., horizon, action_dim) -- the full H-length action chunk.

        Returns:
            Dict[int, Array] mapping each h in chunk_sizes to Q_h, shape (...,) matching observations'
            batch shape. A dict (not a stacked array) because chunk_sizes need not be contiguous or
            evenly spaced, so there's no natural axis to stack along without re-deriving h from index.
        """
        q_all = self.all_positions(observations, actions)
        return {h: q_all[..., h] for h in self.chunk_sizes}