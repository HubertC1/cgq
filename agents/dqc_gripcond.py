"""DQC variant: two independent critics (no distillation), V bootstrapped from whichever one
matches the current state's stochasticity regime.

Independent agent file (agents/dqc.py is left untouched). Q_phi (the h-step chunk critic) and
Q^P_psi (the h_a-step action critic) are both trained by their own direct TD loss against the same
shared V, rather than Q^P_psi being distilled from Q_phi:

  L(phi)   = (Q_phi(s_t, a_t:t+h)     - sum_{k=0}^{h-1}  gamma^k r_{t+k} - gamma^h  V_bar(s_t+h))^2
  L(psi)   = (Q^P_psi(s_t, a_t:t+ha)  - sum_{k=0}^{ha-1} gamma^k r_{t+k} - gamma^ha V_bar(s_t+ha))^2

(dqc.py's own distillation loss, L(psi) = f_kappa_d_expectile(Q_bar_phi - Q^P_psi), does not exist
in this variant.)

The V update then picks its bootstrap source PER-STATE based on gripper closedness -- the direct,
cheap instantiation of CLAUDE.md's Phase-2 "state-dependent backup depth" idea: grasp is where the
slip hazard actually lives (cube_slip_oracle.py only ever fires while the gripper is closed), so
gripping states get the short h_a-step critic and reaching states get the long h-step critic:

  ex_q(s_t) = Q^P_psi_bar(s_t, a_t:t+ha)  if gripper_opening(s_t) > gripper_closed_threshold
            = Q_phi_bar(s_t, a_t:t+h)     otherwise

Both Q_bar terms are read from target networks (target_chunk_critic / target_action_critic), not
just stop-gradient of the live nets. This is a deliberate extension of dqc.py's OWN existing
precedent, not an invented convention: in dqc.py, action_critic already has a target
(target_action_critic) used specifically to source the V-update's bootstrap (see dqc.py's
action_critic_loss). Now that BOTH critics feed V, both get that same treatment -- chunk_critic
gains a target_chunk_critic it doesn't have in dqc.py or dqc_nodistill.py.

V ALSO now has a target (target_value), used by chunk_critic_loss/action_critic_td_loss's own
bootstrap in place of the live value net. This goes beyond what either dqc.py or agents/iql.py do
-- neither has a target_value; both bootstrap their (single) Q from the live V and rely on exactly
one lagged return edge (Q's target feeding V) to keep the V<->Q loop stable. That's not a precedent
for omitting target_value here, though -- it's a precedent that assumes exactly one loop touching
V. This variant has TWO independent TD loops sharing the same V (chunk_critic's and
action_critic's, mixed per-sample by gripper state), a topology neither dqc.py nor iql.py has, so
their single-lag solution isn't guaranteed to carry over. target_value adds a second lag point (on
the V->Q forward edge, alongside the existing Q->V lag), so every edge in both loops is now damped.
Added specifically because the first version of this file (only the V->Q edges live, Q->V edges
lagged) diverged in training -- q_min blew up to O(600) instead of the O(-100) every other
baseline/variant saw. See conversation history for the full diagnosis.

gripper_opening is read directly out of batch['observations'] at a fixed index -- manipspace_env.py's
compute_observation() concatenates [joint_pos(6), joint_vel(6), effector_pos(3), cos_yaw(1),
sin_yaw(1), gripper_opening*gripper_scaler(1), gripper_contact(1), ...block info], so index 17,
divided back out of gripper_scaler=3.0. Uses this codebase's existing "closed" convention
(cube_slip_oracle.py, envs/slip_wrapper.py): gripper_opening > gripper_closed_threshold (default
0.5, matching those files' own default). This makes the agent cube-family-specific by construction
-- gripper_opening_obs_idx/gripper_opening_scaler are config fields (not hardcoded in the loss) so
a different env layout can override them, but the defaults only make sense for cube-* envs.
"""
import copy
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


class DQCGripCondAgent(flax.struct.PyTreeNode):
    """Decoupled Q-chunking, dual-independent-critic variant with gripper-conditional V backup."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def chunk_critic_loss(self, batch, grad_params, rng):
        """Q_phi's own h-step TD loss. Identical to DQCAgent.chunk_critic_loss."""
        rng, _ = jax.random.split(rng)

        batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))

        next_v = self.network.select('target_value')(batch['next_observations'][..., -1, :])

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

    def action_critic_td_loss(self, batch, grad_params, rng):
        """Q^P_psi's own INDEPENDENT h_a-step TD loss -- not distilled from Q_phi. Same structural
        form as chunk_critic_loss, just indexed at policy_chunk_size-1 (h_a steps ahead) instead of
        -1 (h steps ahead); utils/datasets.py's sample_sequence already gives rewards/masks as
        running cumulative/sticky sums indexed by within-window step, so this is a direct slice,
        not a re-derivation."""
        rng, _ = jax.random.split(rng)
        ha = self.config['policy_chunk_size']

        ha_actions = jnp.reshape(batch['actions'], (batch['actions'].shape[0], -1))[..., :self.config['ac_action_dim']]

        next_v = self.network.select('target_value')(batch['next_observations'][..., ha - 1, :])
        target_v = batch['rewards'][..., ha - 1] + \
            (self.config['discount'] ** ha) * batch['masks'][..., ha - 1] * next_v

        q = self.network.select('action_critic')(batch['observations'], actions=ha_actions, params=grad_params)
        critic_loss = (jnp.square(q - target_v) * batch['valid'][..., ha - 1]).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def value_loss(self, batch, grad_params, rng):
        """V update, bootstrapping from whichever critic's TARGET network matches the current
        state's grip status. See module docstring for why target_chunk_critic/target_action_critic
        (not stop-gradient reads of the live nets) are used here."""
        chunk_actions = jnp.reshape(batch['actions'], (batch['actions'].shape[0], -1))
        ha_actions = chunk_actions[..., :self.config['ac_action_dim']]

        ex_qs_h = self.network.select('target_chunk_critic')(batch['observations'], actions=chunk_actions)
        ex_qs_ha = self.network.select('target_action_critic')(batch['observations'], actions=ha_actions)
        if self.config['q_agg'] == "mean":
            ex_q_h = ex_qs_h.mean(axis=0)
            ex_q_ha = ex_qs_ha.mean(axis=0)
        else:
            ex_q_h = ex_qs_h.min(axis=0)
            ex_q_ha = ex_qs_ha.min(axis=0)

        gripper_opening = batch['observations'][..., self.config['gripper_opening_obs_idx']] / self.config['gripper_opening_scaler']
        is_gripping = gripper_opening > self.config['gripper_closed_threshold']
        ex_q = jnp.where(is_gripping, ex_q_ha, ex_q_h)

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
            'frac_gripping': is_gripping.mean(),  # diagnostic: is the switch actually firing sensibly
        }

    def actor_loss(self, batch, grad_params, rng):
        # batch['actions'] : (batch_size, horizon_length, action_dim). Identical to
        # DQCAgent.actor_loss -- BC-regresses the h_a-step slice (ac_action_dim ==
        # policy_chunk_size * action_dim).
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

        rng, actor_rng, value_rng, action_critic_rng, chunk_critic_rng = jax.random.split(rng, 5)

        chunk_critic_loss, chunk_critic_info = self.chunk_critic_loss(batch, grad_params, chunk_critic_rng)
        for k, v in chunk_critic_info.items():
            info[f'chunk_critic/{k}'] = v

        action_critic_loss, action_critic_info = self.action_critic_td_loss(batch, grad_params, action_critic_rng)
        for k, v in action_critic_info.items():
            info[f'action_critic/{k}'] = v

        value_loss, value_info = self.value_loss(batch, grad_params, value_rng)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = chunk_critic_loss + action_critic_loss + value_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @staticmethod
    def _update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'chunk_critic')
        self.target_update(new_network, 'action_critic')
        self.target_update(new_network, 'value')

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
        # Unchanged from DQCAgent: samples h_a-step chunks from actor_bc, scores with
        # action_critic -- dimensionally still correct here since action_critic is a genuine
        # h_a-step critic in this variant too (just independently trained, not distilled).
        del temperature
        seed = rng if rng is not None else self.rng

        def sample_fn(key):
            noises = jax.random.normal(key, (*observations.shape[: -len(self.config['ob_dims'])], self.config['ac_action_dim']))
            actions = self.compute_flow_actions(observations, noises)
            return actions

        def score_fn(actions):
            if self.config["q_agg"] == "mean":
                q = self.network.select("action_critic")(observations, actions=actions).mean(axis=0)
            elif self.config["q_agg"] == "min":
                q = self.network.select("action_critic")(observations, actions=actions).min(axis=0)
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
        assert config['policy_chunk_size'] <= config['horizon_length'], (
            f"policy_chunk_size (h_a={config['policy_chunk_size']}) must be <= horizon_length "
            f"(h={config['horizon_length']}) -- action_critic_td_loss indexes batch windows "
            f"(sampled at sequence_length=horizon_length) at position policy_chunk_size-1, which "
            f"must fall within that window."
        )

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_action_chunks = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        ac_action_dim = config["policy_chunk_size"] * action_dim
        ex_action_low_chunks = ex_action_chunks[..., :ac_action_dim]

        # Define networks. Both chunk_critic and action_critic get a target this time (see module
        # docstring): both now feed V's bootstrap, matching dqc.py's own precedent of sourcing
        # V's bootstrap from a target network, not a stop-gradient of the live one.
        chunk_critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_qs'],
        )
        target_chunk_critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_qs'],
        )

        value_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=1,
        )
        target_value_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=1,
        )

        action_critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_qs'],
        )
        target_action_critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_qs'],
        )

        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=ac_action_dim,
            layer_norm=config['actor_layer_norm'],
        )

        network_info = dict(
            chunk_critic=(chunk_critic_def, (ex_observations, ex_action_chunks)),
            target_chunk_critic=(target_chunk_critic_def, (ex_observations, ex_action_chunks)),
            action_critic=(action_critic_def, (ex_observations, ex_action_low_chunks)),
            target_action_critic=(target_action_critic_def, (ex_observations, ex_action_low_chunks)),
            value=(value_def, (ex_observations,)),
            target_value=(target_value_def, (ex_observations,)),
            actor_bc=(actor_bc_flow_def, (ex_observations, ex_action_low_chunks, ex_times)),  # unconditional BC
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_chunk_critic'] = params['modules_chunk_critic']
        params['modules_target_action_critic'] = params['modules_action_critic']
        params['modules_target_value'] = params['modules_value']

        config['ob_dims'] = ob_dims
        config["action_dim"] = action_dim
        config["ac_action_dim"] = ac_action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='dqc_gripcond',  # Agent name.
            lr=3e-4,            # Learning rate.

            ob_dims=ml_collections.config_dict.placeholder(list),   # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int), # Action dimension (will be set automatically).

            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Policy network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,        # Whether to use layer normalization for the critic(s).
            actor_layer_norm=True,  # Whether to use layer normalization for the policy.

            discount=0.999, # Discount factor.
            tau=0.005,      # Target network update rate (chunk_critic AND action_critic targets).
            num_qs=2,       # Number of Q ensembles.
            q_agg='mean',   # Aggregation function for Q values
            flow_steps=10,  # Number of flow steps for the policy.

            horizon_length=10,          # h. Chunk critic Q_phi's backup horizon.
            policy_chunk_size=1,        # h_a. Action critic Q^P_psi's backup horizon. Must be <= h.

            implicit_backup_type="quantile",    # Implicit maximization loss for the V update
            kappa_b=0.9,                        # Implicit value-backup coefficient

            # Gripper-conditional V-backup switch (cube-family envs only -- see module docstring
            # for the manipspace_env.py observation-layout provenance of these indices/defaults).
            gripper_opening_obs_idx=17,
            gripper_opening_scaler=3.0,
            gripper_closed_threshold=0.5,

            best_of_n=32,                       # Best-of-N policy extraction (h_a-step chunks)
        )
    )
    return config
