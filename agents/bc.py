from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Actor


class BCAgent(flax.struct.PyTreeNode):
    """Plain behavior-cloning baseline: actor only, no critic, no value function.

    CLAUDE.md's Phase 1 is a value-learning study -- this agent has no value-learning objective at
    all, so it isn't one of the value-learning baselines being compared. It exists as the floor:
    the return achieved by literally replaying the dataset's behavior policy, closed-loop (replans
    every env step, single-step actions -- no chunking, regardless of --horizon_length). Every
    value-learning method's whole point is to beat this floor via stitching / in-sample improvement;
    sarsa.py's V^{pi_beta} is the same floor measured as a value rather than a rollout return, so
    the two should roughly agree if both are working.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def actor_loss(self, batch, grad_params):
        """Unweighted BC log-prob on the dataset's single-step action."""
        actions = batch['actions'][..., 0, :]
        dist = self.network.select('actor')(batch['observations'], params=grad_params)
        log_prob = dist.log_prob(actions)
        actor_loss = -(log_prob * batch['valid'][..., -1]).mean()
        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_log_prob': log_prob.mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        actor_loss, actor_info = self.actor_loss(batch, grad_params)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v
        return actor_loss, info

    @staticmethod
    def _update(agent, batch):
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
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
            encoders['actor'] = encoder_module()

        # Define networks.
        actor_def = Actor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            state_dependent_std=False,
            const_std=config['const_std'],
            encoder=encoders.get('actor'),
        )

        network_info = dict(
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

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='bc',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            const_std=True,  # Whether the actor uses a fixed (vs. learned) standard deviation.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, ...).
            horizon_length=ml_collections.config_dict.placeholder(int),  # Unused -- this agent always
            # trains/rolls out single-step, closed-loop, regardless of --horizon_length (same
            # reasoning as main.py's fql special-case). Kept only because main.py unconditionally
            # writes `config['horizon_length'] = FLAGS.horizon_length` before agent.create().
            weight_decay=0.,  # Weight decay.
        )
    )
    return config
