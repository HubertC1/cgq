from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Actor, ActorVectorField
from utils.flow_policy import flow_bc_loss, compute_flow_actions


class BCAgent(flax.struct.PyTreeNode):
    """Plain behavior-cloning baseline: actor only, no critic, no value function.

    CLAUDE.md's Phase 1 is a value-learning study -- this agent has no value-learning objective at
    all, so it isn't one of the value-learning baselines being compared. Two policy_method modes:

    - 'gaussian' (default): the original floor baseline. Replays the dataset's behavior policy,
      closed-loop (replans every env step, single-step actions -- no chunking, regardless of
      horizon_length). Every value-learning method's whole point is to beat this floor via
      stitching / in-sample improvement; sarsa.py's V^{pi_beta} is the same floor measured as a
      value rather than a rollout return, so the two should roughly agree if both are working.
    - 'flow': trains a flow-matching BC policy over the flattened horizon_length-chunk (see
      utils/flow_policy.py). This is NOT a value-learning baseline either, and it is deliberately
      NOT bundled with any critic's rejection sampling (unlike iql.py's/aciql.py's own
      'flow_rejection' policy_method) -- it exists to be trained *once*, frozen, and shared as an
      external, fixed proposal distribution across every critic's rejection-sampling eval, so the
      eval-time policy extraction is held fixed across baselines per CLAUDE.md's Phase 1
      methodology, not merely "trained the same way" per baseline (which would leave each baseline
      with its own independently-SGD'd copy -- a confound). One frozen checkpoint per
      (horizon_length, dataset) combination that appears in the sweep.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def actor_loss(self, batch, grad_params):
        """Unweighted BC log-prob on the dataset's single-step action. policy_method='gaussian'."""
        actions = batch['actions'][..., 0, :]
        dist = self.network.select('actor')(batch['observations'], params=grad_params)
        log_prob = dist.log_prob(actions)
        actor_loss = -(log_prob * batch['valid'][..., -1]).mean()
        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_log_prob': log_prob.mean(),
        }

    def flow_actor_loss(self, batch, grad_params, rng):
        """Flow-matching BC loss (linear-path) over the flattened horizon_length-chunk.
        policy_method='flow'. Pure behavior cloning -- no critic anywhere in this loss, so no
        target network is needed (nothing bootstraps). See utils/flow_policy.py.
        """
        chunk_actions = jnp.reshape(batch['actions'], (batch['actions'].shape[0], -1))
        actor_bc_flow_fn = lambda o, x, t: self.network.select('actor_bc_flow')(o, x, t, params=grad_params)
        loss = flow_bc_loss(actor_bc_flow_fn, batch['observations'], chunk_actions, batch['valid'][..., -1], rng)
        return loss, {'actor_loss': loss}

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        if self.config['policy_method'] == 'flow':
            actor_loss, actor_info = self.flow_actor_loss(batch, grad_params, rng)
        else:
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
        """policy_method='gaussian': single-step action. policy_method='flow': a full
        horizon_length-chunk, flattened -- matches aciql.py's output convention, so the shared eval
        loop's chunk-unpacking (reshape(-1, action_dim) into an action queue) works unchanged. Note
        this is plain Euler-integration BC sampling, no rejection -- there is no critic here to
        reject against; that happens externally once this checkpoint is loaded alongside one.
        """
        if self.config['policy_method'] == 'flow':
            full_action_dim = self.config['action_dim'] * self.config['horizon_length']
            actor_bc_flow_fn = lambda o, x, t: self.network.select('actor_bc_flow')(o, x, t)
            noises = jax.random.normal(rng, (*observations.shape[:-1], full_action_dim))
            return compute_flow_actions(actor_bc_flow_fn, observations, noises, flow_steps=self.config['flow_steps'])
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
        full_actions = jnp.concatenate([ex_actions] * config['horizon_length'], axis=-1)
        full_action_dim = full_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            if config['policy_method'] == 'flow':
                encoders['actor_bc_flow'] = encoder_module()
            else:
                encoders['actor'] = encoder_module()

        # Define networks.
        network_info = dict()
        if config['policy_method'] == 'flow':
            actor_bc_flow_def = ActorVectorField(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=full_action_dim,
                layer_norm=False,  # explicit -- flow BC policy is never layer-normed
                encoder=encoders.get('actor_bc_flow'),
            )
            ex_times = ex_actions[..., :1]
            network_info['actor_bc_flow'] = (actor_bc_flow_def, (ex_observations, full_actions, ex_times))
        elif config['policy_method'] == 'gaussian':
            actor_def = Actor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                layer_norm=config['actor_layer_norm'],
                state_dependent_std=False,
                const_std=config['const_std'],
                encoder=encoders.get('actor'),
            )
            network_info['actor'] = (actor_def, (ex_observations,))
        else:
            raise ValueError(f"Unknown policy_method: {config['policy_method']!r}")

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
            horizon_length=ml_collections.config_dict.placeholder(int),  # Unused when
            # policy_method='gaussian' (always single-step, closed-loop, regardless of
            # --horizon_length -- same reasoning as main.py's fql special-case). Meaningful when
            # policy_method='flow': defines the BC chunk length, which must match whichever
            # critic's horizon_length this checkpoint is meant to be paired with at eval time.
            weight_decay=0.,  # Weight decay.
            policy_method='gaussian',  # 'gaussian' | 'flow'. 'flow' trains a flow-matching BC
            # policy over the flattened horizon_length-chunk instead of the single-step Gaussian
            # actor, meant to be trained once and frozen, then shared externally across critics'
            # rejection-sampling eval (see class docstring) -- NOT bundled with a critic here.
            flow_steps=10,  # Euler integration steps for the flow BC policy (policy_method='flow' only).
        )
    )
    return config
