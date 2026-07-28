import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Actor, Value


class IQLAgent(flax.struct.PyTreeNode):
    """Implicit Q-learning (IQL) agent (Kostrikov, Nair, Levine 2021).

    Single-task, state-based, no action chunking. V(s) is expectile-regressed against Q(s,a) using
    only dataset actions; Q(s,a) TD-regresses against r + discount * V(s'), bootstrapping over V
    (never over Q or a sampled action) -- the "in-sample max" that makes IQL immune to OOD-action
    extrapolation. This is CLAUDE.md's baseline #1: single-step Bellman backup, full horizon cost,
    immune to both suboptimal-data and stochastic-dynamics bias by construction.

    The actor (AWR) is here only for completeness / eval rollouts -- per CLAUDE.md's Phase 1 design
    the eval-time policy should eventually be the shared flow-matching + rejection-sampling
    extraction used across all baselines, not each agent's own actor. That shared extraction isn't
    built yet, so `sample_actions` below uses this agent's own AWR actor directly, same as acfql.py
    currently does for FQL/QC-FQL.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def value_loss(self, batch, grad_params):
        """V(s) expectile-regressed against target_critic(s, a) using dataset actions only."""
        actions = batch['actions'][..., 0, :]
        qs = self.network.select('target_critic')(batch['observations'], actions=actions)
        q = jnp.min(qs, axis=0) if self.config['q_agg'] == 'min' else jnp.mean(qs, axis=0)
        v = self.network.select('value')(batch['observations'], params=grad_params)
        adv = q - v
        value_loss = (self.expectile_loss(adv, adv, self.config['expectile']) * batch['valid'][..., -1]).mean()
        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def critic_loss(self, batch, grad_params):
        """Q(s, a) TD-regressed against r + discount * V(s'). Bootstraps over V only."""
        next_v = self.network.select('value')(batch['next_observations'][..., -1, :])
        target_q = batch['rewards'][..., -1] + (
            self.config['discount'] ** self.config['horizon_length']
        ) * batch['masks'][..., -1] * next_v
        actions = batch['actions'][..., 0, :]
        qs = self.network.select('critic')(batch['observations'], actions=actions, params=grad_params)
        critic_loss = (jnp.square(qs - target_q[None]) * batch['valid'][..., -1][None]).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': qs.mean(),
            'q_max': qs.max(),
            'q_min': qs.min(),
        }

    def actor_loss(self, batch, grad_params):
        """AWR: BC log-prob weighted by exp(alpha * advantage), clipped for stability.

        Note: `alpha` here is the AWR inverse-temperature (higher = more greedy toward Q) --
        opposite direction from FQL's `alpha`, which is a BC-strength coefficient (higher = more
        conservative). Same config field name for CLI consistency across agents; different agent,
        different sign of effect.
        """
        actions = batch['actions'][..., 0, :]
        v = self.network.select('value')(batch['observations'])
        qs = self.network.select('target_critic')(batch['observations'], actions=actions)
        q = jnp.min(qs, axis=0) if self.config['q_agg'] == 'min' else jnp.mean(qs, axis=0)
        adv = q - v

        exp_a = jnp.exp(adv * self.config['alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        dist = self.network.select('actor')(batch['observations'], params=grad_params)
        log_prob = dist.log_prob(actions)

        actor_loss = -(exp_a * log_prob * batch['valid'][..., -1]).mean()
        return actor_loss, {
            'actor_loss': actor_loss,
            'adv_mean': adv.mean(),
            'bc_log_prob': log_prob.mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = value_loss + critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @staticmethod
    def _update(agent, batch):
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, 'critic')
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)

    @jax.jit
    def batch_update(self, batch):
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    @jax.jit
    def sample_actions(self, observations, rng=None, temperature=1.0):
        dist = self.network.select('actor')(observations, temperature=temperature)
        actions = dist.sample(seed=rng)
        actions = jnp.clip(actions, -1, 1)
        return actions

    @jax.jit
    def get_value(self, observations):
        """V(s). Exposed directly for diagnostics (e.g. rendering a value map over the maze)."""
        return self.network.select('value')(observations)

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['value'] = encoder_module()
            encoders['critic'] = encoder_module()
            encoders['actor'] = encoder_module()

        # Define networks.
        value_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=1,
            encoder=encoders.get('value'),
        )
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_qs'],
            encoder=encoders.get('critic'),
        )
        actor_def = Actor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            state_dependent_std=False,
            const_std=config['const_std'],
            encoder=encoders.get('actor'),
        )

        network_info = dict(
            value=(value_def, (ex_observations,)),
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor=(actor_def, (ex_observations,)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        if config['weight_decay'] > 0.:
            network_tx = optax.adamw(learning_rate=config['lr'], weight_decay=config['weight_decay'])
        else:
            network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='iql',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value/critic network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization for value/critic.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target critic update rate.
            q_agg='mean',  # Aggregation method for the critic ensemble.
            alpha=10.0,  # AWR inverse temperature (higher = more greedy toward Q). NOT the same
            # direction as FQL's `alpha` (BC-strength coefficient, higher = more conservative).
            expectile=0.9,  # IQL expectile for the V regression (higher = closer to in-sample max).
            num_qs=2,  # Critic ensemble size.
            const_std=True,  # Whether the actor uses a fixed (vs. learned) standard deviation.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, ...).
            horizon_length=ml_collections.config_dict.placeholder(int),  # Set by main.py; use 1 for true IQL.
            weight_decay=0.,  # Weight decay.
        )
    )
    return config
