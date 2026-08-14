"""DQC variant: no action-critic distillation, h_a == h.

Independent agent file (agents/dqc.py is left untouched). Differs from DQCAgent in exactly one
structural way: the distilled action critic Q^P_psi is removed entirely, and
config.value.policy_chunk_size is forced to equal config.value.horizon_length (h_a = h) --
create() asserts this.

What changes mechanically:
  - chunk_critic_loss (Q_phi's own h-step TD loss): UNCHANGED, copied verbatim from dqc.py.
  - action_critic_loss's distillation term (L(psi) = f_kappa_d_expectile(Q_bar_phi - Q^P_psi)): GONE.
    There is no action_critic network at all in this variant.
  - value_loss: V now bootstraps directly from Q_phi (read via its own live params, no gradient --
    matches dqc.py's existing Q_bar_phi convention of "stop-gradient on the live chunk_critic",
    since dqc.py has never had a target_chunk_critic either) instead of from a distilled
    target_action_critic. Same kappa_b-weighted quantile/expectile form as dqc.py.
  - actor_loss: UNCHANGED code, but now BC-regresses a FULL h-step action chunk (ac_action_dim ==
    horizon_length * action_dim, since policy_chunk_size == horizon_length here) instead of a
    short h_a-step slice.
  - sample_actions: "best-of-N" becomes "best of N h-step chunks" -- score_fn now scores candidate
    chunks with chunk_critic directly (there is no action_critic to score with). Chunks longer than
    1 step are already handled generically by evaluation.py's rollout loop (it reshapes whatever
    sample_actions returns into (-1, action_dim) and steps through it), so no eval-side change
    needed.
  - No target-network machinery at all: value has never had a target in DQC, and with
    action_critic gone there's nothing left that had one either. target_update becomes a no-op.
"""
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from functools import partial

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Value, ActorVectorField


def apply_bfn(sample_fn, score_fn, n):
    def fn(rng):
        y = jax.vmap(sample_fn)(jax.random.split(rng, n))
        scores = jax.vmap(score_fn)(y)
        indices = jnp.argmax(scores, axis=0)
        y_reshaped = y.reshape((n, -1, y.shape[-1]))
        batch_size = y_reshaped.shape[1]
        indices_reshaped = indices.reshape(-1)
        y_out = y_reshaped[indices_reshaped, jnp.arange(batch_size)].reshape((y.shape[1:]))
        return y_out
    return fn


class DQCNoDistillAgent(flax.struct.PyTreeNode):
    """Decoupled Q-chunking, no-distillation variant (h_a = h)."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def chunk_critic_loss(self, batch, grad_params, rng):
        """Compute Q_phi's h-step TD loss. Identical to DQCAgent.chunk_critic_loss."""
        rng, _ = jax.random.split(rng)

        batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))

        next_v = self.network.select('value')(batch['next_observations'][..., -1, :])

        target_v = batch['rewards'][..., -1] + \
            (self.config['discount'] ** self.config['horizon_length']) * batch['masks'][..., -1] * next_v
        q = self.network.select('chunk_critic')(
            batch['observations'],
            actions=batch_actions, params=grad_params)
        critic_loss = (jnp.square(q - target_v) * batch['valid'][..., -1]).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def value_loss(self, batch, grad_params, rng):
        """V update, bootstrapped directly from Q_bar_phi (no distilled action critic)."""
        batch_chunk_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))

        # Q_bar_phi(s_t, a_t:t+h) -- read via live (non-grad_params) params, i.e. stop-gradient.
        # Matches dqc.py's own convention: chunk_critic has never had a separate target network.
        ex_qs = self.network.select('chunk_critic')(batch['observations'], actions=batch_chunk_actions)
        if self.config['q_agg'] == "mean":
            ex_q = ex_qs.mean(axis=0)
        else:
            ex_q = ex_qs.min(axis=0)

        v = self.network.select('value')(batch['observations'], params=grad_params)

        if self.config["implicit_backup_type"] == "expectile":
            weight = jnp.where(ex_q >= v, self.config['kappa_b'], (1 - self.config['kappa_b']))
            value_loss = (weight * jnp.square(v - ex_q) * batch['valid'][..., -1]).mean()
        elif self.config["implicit_backup_type"] == "quantile":
            weight = jnp.where(ex_q >= v, self.config['kappa_b'], (1 - self.config['kappa_b']))
            value_loss = (weight * jnp.abs(v - ex_q) * batch['valid'][..., -1]).mean()
        else:
            raise NotImplementedError

        return value_loss, {
            'value_loss': value_loss,
            'adv': (ex_q - v).mean(),
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        # batch['actions'] : (batch_size, horizon_length, action_dim). Identical to
        # DQCAgent.actor_loss -- ac_action_dim now equals horizon_length * action_dim (h_a == h),
        # so this BC-regresses the full h-step chunk rather than a short h_a-step slice.
        batch_size, _, _ = batch['actions'].shape
        rng, x_rng, t_rng, _ = jax.random.split(rng, 4)

        x_0 = jax.random.normal(x_rng, (batch_size, self.config["ac_action_dim"]))
        x_1 = batch['actions'].reshape(batch_size, -1)[..., :self.config["ac_action_dim"]]
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_bc')(batch['observations'], actions=x_t, times=t, params=grad_params)
        bc_flow_loss = jnp.mean(jnp.mean(jnp.square(pred - vel), axis=-1) * batch["valid"][..., -1])

        return bc_flow_loss, {"bc_flow_loss": bc_flow_loss}

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, value_rng, chunk_critic_rng = jax.random.split(rng, 4)

        chunk_critic_loss, chunk_critic_info = self.chunk_critic_loss(batch, grad_params, chunk_critic_rng)
        for k, v in chunk_critic_info.items():
            info[f'chunk_critic/{k}'] = v

        value_loss, value_info = self.value_loss(batch, grad_params, value_rng)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = chunk_critic_loss + value_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """No-op: no network in this variant has a target counterpart (value never has, and
        action_critic -- the only thing dqc.py target-updates -- doesn't exist here)."""
        del network, module_name

    @staticmethod
    def _update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)

    @jax.jit
    def batch_update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    @partial(jax.jit, static_argnames="best_of_n_override")
    def sample_actions(
        self,
        observations,
        rng=None,
        best_of_n_override=None,
        temperature=1.0,
    ):
        # temperature is accepted but unused (see DQCAgent.sample_actions's identical note).
        del temperature
        seed = rng if rng is not None else self.rng

        def sample_fn(key):
            noises = jax.random.normal(key, (*observations.shape[: -len(self.config['ob_dims'])], self.config['ac_action_dim']))
            actions = self.compute_flow_actions(observations, noises)
            return actions

        def score_fn(actions):
            # "Best of N chunks": score_fn now scores whole h-step chunks with chunk_critic --
            # there is no action_critic to score with in this variant, and ac_action_dim ==
            # chunk_critic's own action dimensionality (h_a == h), so this is a direct swap.
            if self.config["q_agg"] == "mean":
                q = self.network.select("chunk_critic")(observations, actions=actions).mean(axis=0)
            elif self.config["q_agg"] == "min":
                q = self.network.select("chunk_critic")(observations, actions=actions).min(axis=0)
            return q

        bfn_sample_fn = apply_bfn(sample_fn, score_fn, self.config["best_of_n"] if best_of_n_override is None else best_of_n_override)
        return bfn_sample_fn(seed)

    @jax.jit
    def compute_flow_actions(
        self,
        observations,
        noises,
    ):
        actions = noises
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select("actor_bc")(observations, actions=actions, times=t, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
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
            ex_observations: Example observations.
            ex_actions: Example actions.
            config: Configuration dictionary.
        """
        assert config['policy_chunk_size'] == config['horizon_length'], (
            f"DQCNoDistillAgent requires h_a == h (policy_chunk_size == horizon_length), got "
            f"policy_chunk_size={config['policy_chunk_size']} != horizon_length={config['horizon_length']}. "
            f"This is the defining assumption of the no-distillation variant -- if you want h_a != h, "
            f"use DQCAgent (agents/dqc.py) instead."
        )

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_action_chunks = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        ac_action_dim = config["policy_chunk_size"] * action_dim  # == horizon_length * action_dim here

        # Define networks. No action_critic/target_action_critic in this variant.
        chunk_critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_qs'],
        )

        value_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=1,
        )

        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=ac_action_dim,
            layer_norm=config['actor_layer_norm'],
        )

        network_info = dict(
            chunk_critic=(chunk_critic_def, (ex_observations, ex_action_chunks)),
            value=(value_def, (ex_observations,)),
            actor_bc=(actor_bc_flow_def, (ex_observations, ex_action_chunks, ex_times)),  # unconditional BC
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        config['ob_dims'] = ob_dims
        config["action_dim"] = action_dim
        config["ac_action_dim"] = ac_action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='dqc_nodistill',  # Agent name.
            lr=3e-4,            # Learning rate.

            ob_dims=ml_collections.config_dict.placeholder(list),   # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int), # Action dimension (will be set automatically).

            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Policy network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,        # Whether to use layer normalization for the critic(s).
            actor_layer_norm=True,  # Whether to use layer normalization for the policy.

            discount=0.999, # Discount factor.
            num_qs=2,       # Number of Q ensembles.
            q_agg='mean',   # Aggregation function for Q values
            flow_steps=10,  # Number of flow steps for the policy.

            # Horizon: h_a == h is enforced in create().
            horizon_length=10,          # h. Chunk critic's backup horizon AND policy chunk size.
            policy_chunk_size=10,       # h_a. Must equal horizon_length -- create() asserts this.

            implicit_backup_type="quantile",    # Implicit maximization loss for the V update
            kappa_b=0.9,                        # Implicit value-backup coefficient

            best_of_n=32,                       # Best-of-N-chunks policy extraction
        )
    )
    return config
