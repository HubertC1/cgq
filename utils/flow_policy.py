"""Shared flow-matching BC policy + rejection-sampling extraction, usable by any value-learning
agent (iql.py, aciql.py, ...) as an alternative to AWR. Lifted out of agents/acfql.py's
"best-of-n" actor_type, which already implements this exact mechanism -- but bundled there with
FQL's own critic (which bootstraps through a sampled action). Here the two are deliberately kept
separate: the caller's critic_loss/value_loss are untouched (in-sample expectile regression,
never bootstrapping through a sampled action), and only *policy extraction* -- how an action gets
proposed given a learned Q -- changes. This is CLAUDE.md's Phase 1 target: "flow-matching policy
with rejection sampling against the learned value," held fixed across value-learning baselines
while only the value objective varies.

Mechanism:
  - Train a flow velocity field v_theta(t, s, x_t) via the linear-path flow-matching BC loss:
    x_0 ~ N(0,I), x_1 = dataset action (chunk), t ~ Uniform(0,1), x_t = (1-t)x_0 + t*x_1,
    regress v_theta(t,s,x_t) toward (x_1 - x_0). Pure behavior cloning -- no Q anywhere in this
    loss, so no target network is needed (nothing here bootstraps).
  - At inference, sample N candidates from the flow policy via Euler integration starting from
    Gaussian noise, evaluate the caller's own (already-trained) critic at each candidate, and
    return the argmax-Q candidate per state (rejection sampling).
"""

import jax
import jax.numpy as jnp

from utils.flax_utils import restore_agent_with_file


def flow_bc_loss(actor_bc_flow_fn, observations, chunk_actions, valid, rng):
    """Linear-path flow-matching BC loss.

    Args:
        actor_bc_flow_fn: callable (observations, x_t, t) -> predicted velocity. Should already be
            bound to the correct module/params, e.g.
            `lambda o, x, t: network.select('actor_bc_flow')(o, x, t, params=grad_params)`.
        observations: (batch, obs_dim) states.
        chunk_actions: (batch, action_dim) flattened target actions/chunks (x_1).
        valid: (batch,) mask, 1 for valid samples (e.g. batch['valid'][..., -1]).
        rng: PRNG key for x_0 and t sampling.

    Returns:
        Scalar loss.
    """
    batch_size, action_dim = chunk_actions.shape
    x_rng, t_rng = jax.random.split(rng)
    x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
    x_1 = chunk_actions
    t = jax.random.uniform(t_rng, (batch_size, 1))
    x_t = (1 - t) * x_0 + t * x_1
    vel = x_1 - x_0

    pred = actor_bc_flow_fn(observations, x_t, t)
    per_sample_loss = jnp.mean(jnp.square(pred - vel), axis=-1)
    return (per_sample_loss * valid).mean()


def compute_flow_actions(actor_bc_flow_fn, observations, noises, flow_steps):
    """Euler-integrate the flow from noise to an action sample.

    Args:
        actor_bc_flow_fn: callable (observations, x_t, t) -> predicted velocity.
        observations: (..., obs_dim) states, already broadcast to noises' leading shape.
        noises: (..., action_dim) initial x_0 samples.
        flow_steps: number of Euler integration steps.

    Returns:
        (..., action_dim) actions, clipped to [-1, 1].
    """
    actions = noises
    for i in range(flow_steps):
        t = jnp.full((*actions.shape[:-1], 1), i / flow_steps)
        vels = actor_bc_flow_fn(observations, actions, t)
        actions = actions + vels / flow_steps
    return jnp.clip(actions, -1, 1)


def flow_rejection_sample_actions(
    actor_bc_flow_fn, critic_fn, observations, rng, num_samples, flow_steps, action_dim, q_agg='mean'
):
    """Sample num_samples candidate actions/chunks from the flow BC policy, evaluate each with
    critic_fn, and return the argmax-Q candidate per observation.

    Args:
        actor_bc_flow_fn: callable (observations, x_t, t) -> predicted velocity.
        critic_fn: callable (observations, actions) -> Q values, shape (num_qs, ...) matching this
            project's Value network ensemble convention (ensemble dim first).
        observations: (..., obs_dim) states (a single state or a batch).
        rng: PRNG key.
        num_samples: N, number of candidates per observation.
        flow_steps: Euler integration steps.
        action_dim: flattened action (chunk) dimension.
        q_agg: 'mean' or 'min' -- how to aggregate the critic ensemble before argmax.

    Returns:
        (..., action_dim) selected actions.
    """
    batch_shape = observations.shape[:-1]
    noises = jax.random.normal(rng, (*batch_shape, num_samples, action_dim))
    obs_tiled = jnp.repeat(observations[..., None, :], num_samples, axis=-2)  # (..., num_samples, obs_dim)

    actions = compute_flow_actions(actor_bc_flow_fn, obs_tiled, noises, flow_steps)  # (..., num_samples, action_dim)

    qs = critic_fn(obs_tiled, actions)  # (num_qs, ..., num_samples)
    q = qs.min(axis=0) if q_agg == 'min' else qs.mean(axis=0)  # (..., num_samples)
    indices = jnp.argmax(q, axis=-1)  # (...,)

    out_shape = indices.shape
    flat_indices = indices.reshape(-1)
    bsize = flat_indices.shape[0]
    flat_actions = actions.reshape(-1, num_samples, action_dim)
    selected = flat_actions[jnp.arange(bsize), flat_indices, :]
    return selected.reshape(*out_shape, action_dim)


def load_frozen_bc_flow_params(bc_checkpoint, ex_observations, ex_actions, horizon_length, actor_hidden_dims, seed=0):
    """Load a pretrained agents/bc.py flow-BC checkpoint's actor_bc_flow params, for use as a
    frozen (non-trainable) proposal distribution inside another agent's rejection-sampling policy
    extraction (iql.py/aciql.py's policy_method='flow_rejection').

    Reconstructs a throwaway agents.bc.BCAgent skeleton with the exact architecture the checkpoint
    was trained with (policy_method='flow', same horizon_length/actor_hidden_dims), restores the
    checkpoint into it, and returns just the actor_bc_flow params subtree -- the caller is
    responsible for splicing this into its own network's params (see iql.py/aciql.py's create())
    and for never passing params=grad_params for that module in any loss term, so autodiff assigns
    it an identically-zero gradient and it never departs from these loaded values (same mechanism
    already used for target_critic, just without even a polyak update).

    ex_observations/ex_actions should be the same example batch the caller itself was built from --
    the BC checkpoint must have been trained on the same dataset/env for shapes (and semantics) to
    match.
    """
    import agents.bc as bc_agent_module  # local import: avoids a module-level agents.bc <->
    # agents.{iql,aciql} import cycle (agents/bc.py has no reason to import either of those).

    bc_config = bc_agent_module.get_config()
    bc_config['policy_method'] = 'flow'
    bc_config['horizon_length'] = horizon_length
    bc_config['actor_hidden_dims'] = actor_hidden_dims
    bc_skeleton = bc_agent_module.BCAgent.create(seed, ex_observations, ex_actions, bc_config)
    bc_skeleton = restore_agent_with_file(bc_skeleton, bc_checkpoint)
    return bc_skeleton.network.params['modules_actor_bc_flow']
