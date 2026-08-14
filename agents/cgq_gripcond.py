"""CGQ variant: gripper-conditional anchor-loss weight (beta).

Independent agent file (agents/cgq.py is left untouched). Identical to CGQAgent in every respect
except step_critic_loss's anchor-loss weighting: instead of a single scalar `beta`, the weight is
selected PER-SAMPLE based on gripper closedness, same idea and same gripper-detection mechanism as
agents/dqc_gripcond.py's V-backup switch, applied here to CGQ's anchor loss instead:

  beta(s_t) = beta_gripping   if gripper_opening(s_t) > gripper_closed_threshold
            = beta_reaching   otherwise

beta is the coefficient that anchors the short step_critic toward the long chunk_critic's value
(an expectile-weighted regression of Q_step towards Q_chunk -- see cgq.py's step_critic_loss
docstring, "SHARSA + Expectile Loss from chunking critic"). A LARGE beta while reaching means: lean
heavily on the long-horizon chunk critic's estimate (comparatively deterministic regime, the long
view is trustworthy). A SMALL beta while gripping means: trust the chunk critic much less (the slip
hazard only ever fires while gripping -- cube_slip_oracle.py -- so long-horizon estimates spanning
a grasp are exactly where they're least reliable), and let the step critic rely more on its own
short-horizon TD signal instead. Same underlying design principle as dqc_gripcond.py's state-
dependent backup depth, applied to CGQ's anchor mechanism rather than DQC's V-bootstrap-selection
mechanism.

Mechanically: this must be applied to the PER-SAMPLE anchor loss before it's mean-reduced, not to
the already-averaged scalar anchor_loss -- multiplying a single reduced scalar by one beta value
would silently discard the state-dependence (there'd be nothing left to select on). See
step_critic_loss below: the mean() now happens after beta is applied, not before.

No new networks, no create()/target_update changes -- gripper_opening is read directly out of
batch['observations'] at a fixed index, exactly as in dqc_gripcond.py (see that file's module
docstring for the manipspace_env.py observation-layout provenance of
gripper_opening_obs_idx=17/gripper_opening_scaler=3.0/gripper_closed_threshold=0.5).
"""
import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value

class CGQGripCondAgent(flax.struct.PyTreeNode):


    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def chunk_critic_loss(self, batch, grad_params, rng):
        """Compute the FQL critic loss."""

        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :] # take the first action

        # TD loss
        rng, sample_rng = jax.random.split(rng)
        next_actions = self.sample_chunk_actions(batch['next_observations'][..., -1, :], rng=sample_rng)

        next_qs = self.network.select(f'target_chunk_critic')(batch['next_observations'][..., -1, :], actions=next_actions)
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        target_q = batch['rewards'][..., -1] + \
            (self.config['discount'] ** self.config["horizon_length"]) * batch['masks'][..., -1] * next_q

        q = self.network.select('chunk_critic')(batch['observations'], actions=batch_actions, params=grad_params)

        critic_loss = (jnp.square(q - target_q) * batch['valid'][..., -1]).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def chunk_actor_loss(self, batch, grad_params, rng):
        """Compute the FQL actor loss."""
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))  # fold in horizon_length together with action_dim
        else:
            batch_actions = batch["actions"][..., 0, :] # take the first one
        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC flow loss.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch_actions
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('chunk_actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)

        # only bc on the valid chunk indices
        if self.config["action_chunking"]:
            bc_flow_loss = jnp.mean(
                jnp.reshape(
                    (pred - vel) ** 2,
                    (batch_size, self.config["horizon_length"], self.config["action_dim"])
                ) * batch["valid"][..., None]
            )
        else:
            bc_flow_loss = jnp.mean(jnp.square(pred - vel))

        if self.config["chunk_actor_type"] == "distill-ddpg":
            # Distillation loss.
            rng, noise_rng = jax.random.split(rng)
            noises = jax.random.normal(noise_rng, (batch_size, action_dim))
            target_flow_actions = self.compute_chunk_flow_actions(batch['observations'], noises=noises)
            actor_actions = self.network.select('chunk_actor_onestep_flow')(batch['observations'], noises, params=grad_params)
            distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

            # Q loss.
            actor_actions = jnp.clip(actor_actions, -1, 1)

            qs = self.network.select(f'chunk_critic')(batch['observations'], actions=actor_actions)
            q = jnp.mean(qs, axis=0)
            q_loss = -q.mean()

            if self.config['normalize_q_loss']:
                lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
                q_loss = lam * q_loss
        else:
            distill_loss = jnp.zeros(())
            q_loss = jnp.zeros(())

        # Total loss.
        actor_loss = bc_flow_loss + self.config['alpha'] * distill_loss + q_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'distill_loss': distill_loss,
            'q_loss': q_loss,
        }

    def step_critic_loss(self, batch, grad_params, rng):
        """
        SHARSA + Expectile Loss from chunking critic, with a gripper-conditional (per-sample)
        anchor weight beta instead of CGQAgent's single scalar beta. See module docstring.
        """
        rng, sample_rng = jax.random.split(rng)
        next_obs = batch['next_observations'][..., 0, :]
        batch_rewards = batch['rewards'][..., 0]
        batch_masks = batch['masks'][..., 0]
        actions = batch['actions'][..., 0, :]

        if self.config["step_critic_type"] == "SARSA":
            next_actions = batch['actions'][..., 1, :]
        elif self.config["step_critic_type"] == "actor-critic":
            next_actions = self.sample_actions(next_obs, rng=sample_rng)
        else:
            raise NotImplementedError
        next_qs = self.network.select(f'target_step_critic')(next_obs, actions=next_actions)
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)
        target_q = batch_rewards + self.config['discount'] * batch_masks * next_q

        q = self.network.select('step_critic')(batch['observations'], actions=actions, params=grad_params)
        td_loss = jnp.mean(jnp.square(q - target_q))

        ### Anchor Loss
        batch_actions = jnp.reshape(batch['actions'], (batch['actions'].shape[0], -1))
        ac_values = self.network.select('chunk_critic')(batch['observations'], actions=batch_actions)

        diff = ac_values - q

        # Per-sample anchor loss, NOT yet reduced -- beta must be applied here (per-sample) rather
        # than to an already-averaged scalar, or the state-dependence has nothing left to select.
        per_sample_anchor = self.expectile_loss(diff, diff, self.config['anchor_expectile']) * batch['valid'][..., -1]

        gripper_opening = batch['observations'][..., self.config['gripper_opening_obs_idx']] / self.config['gripper_opening_scaler']
        is_gripping = gripper_opening > self.config['gripper_closed_threshold']
        beta = jnp.where(is_gripping, self.config['beta_gripping'], self.config['beta_reaching'])

        anchor_loss = (beta * per_sample_anchor).mean()
        critic_loss = self.config['td_loss'] * td_loss + anchor_loss

        return critic_loss, {
            'step_critic_loss': critic_loss,
            'td_loss': td_loss,
            'anchor_loss': anchor_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
            'frac_gripping': is_gripping.mean(),
        }

    def step_actor_loss(self, batch, grad_params, rng):
        """ Compute the FQL actor loss. """
        batch_actions = batch['actions'][..., 0, :]
        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC flow loss.
        x_0 = jax.random.normal(x_rng,  (batch_size, action_dim))
        x_1 = batch_actions
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('step_actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)

        bc_flow_loss = jnp.mean((pred - vel) ** 2)

        # Distillation loss.
        if self.config["actor_type"] == "distill-ddpg":
            rng, noise_rng = jax.random.split(rng)
            noises = jax.random.normal(noise_rng, (batch_size, action_dim))
            target_flow_actions = self.compute_step_flow_actions(batch['observations'], noises=noises)
            actor_actions = self.network.select('step_actor_onestep_flow')(batch['observations'], noises, params=grad_params)
            distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

            # Q loss.
            actor_actions = jnp.clip(actor_actions, -1, 1)
            qs = self.network.select('step_critic')(batch['observations'], actions=actor_actions)
            q = jnp.mean(qs, axis=0)

            q_loss = -q.mean()
            if self.config['normalize_q_loss']:
                lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
                q_loss = lam * q_loss
        else:
            distill_loss = jnp.zeros(())
            q_loss = jnp.zeros(())
        # Total loss.
        actor_loss = bc_flow_loss + self.config['step_alpha'] * distill_loss + q_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'distill_loss': distill_loss,
            'q_loss': q_loss,
        }


    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, step_actor_rng, step_critic_rng, chunk_actor_rng, chunk_critic_rng = jax.random.split(rng, 5)

        chunk_critic_loss, chunk_critic_info = self.chunk_critic_loss(batch, grad_params, chunk_critic_rng)
        for k, v in chunk_critic_info.items():
            info[f'chunk_critic/{k}'] = v

        chunk_actor_loss, actor_info = self.chunk_actor_loss(batch, grad_params, chunk_actor_rng)
        for k, v in actor_info.items():
            info[f'chunk_actor/{k}'] = v

        step_critic_loss, step_critic_info = self.step_critic_loss(batch, grad_params, step_critic_rng)
        for k, v in step_critic_info.items():
            info[f'step_critic/{k}'] = v

        step_actor_loss, step_actor_info = self.step_actor_loss(batch, grad_params, step_actor_rng)
        for k, v in step_actor_info.items():
            info[f'step_actor/{k}'] = v


        loss = chunk_critic_loss + chunk_actor_loss + step_critic_loss + step_actor_loss
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
    def _update(agent, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, 'chunk_critic')
        agent.target_update(new_network, 'step_critic')
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)

    @jax.jit
    def batch_update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    @jax.jit
    def sample_chunk_actions(
        self,
        observations,
        rng=None,
    ):
        if self.config["chunk_actor_type"] == "distill-ddpg":
            noises = jax.random.normal(
                rng,
                (
                    *observations.shape[: -len(self.config['ob_dims'])],  # batch_size
                    self.config['action_dim'] * \
                        (self.config['horizon_length'] if self.config["action_chunking"] else 1),
                ),
            )
            actions = self.network.select(f'chunk_actor_onestep_flow')(observations, noises)
            actions = jnp.clip(actions, -1, 1)

        elif self.config["chunk_actor_type"] == "best-of-n":
            action_dim = self.config['action_dim'] * \
                        (self.config['horizon_length'] if self.config["action_chunking"] else 1)
            noises = jax.random.normal(
                rng,
                (
                    *observations.shape[: -len(self.config['ob_dims'])],  # batch_size
                    self.config["chunk_actor_num_samples"], action_dim
                ),
            )
            observations = jnp.repeat(observations[..., None, :], self.config["chunk_actor_num_samples"], axis=-2)
            actions = self.compute_chunk_flow_actions(observations, noises)
            actions = jnp.clip(actions, -1, 1)
            if self.config["q_agg"] == "mean":
                q = self.network.select("chunk_critic")(observations, actions).mean(axis=0)
            else:
                q = self.network.select("chunk_critic")(observations, actions).min(axis=0)
            indices = jnp.argmax(q, axis=-1)

            bshape = indices.shape
            indices = indices.reshape(-1)
            bsize = len(indices)
            actions = jnp.reshape(actions, (-1, self.config["chunk_actor_num_samples"], action_dim))[jnp.arange(bsize), indices, :].reshape(
                bshape + (action_dim,))

        return actions

    @jax.jit
    def compute_chunk_flow_actions(
        self,
        observations,
        noises,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        if self.config['encoder'] is not None:
            observations = self.network.select('chunk_actor_bc_flow_encoder')(observations)
        actions = noises #(B, action_dim)
        # Euler method.
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('chunk_actor_bc_flow')(observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
        return actions

    @jax.jit
    def sample_actions(
        self,
        observations,
        rng=None,
        temperature=1.0,
    ):
        # temperature is accepted but unused: evaluation.py's actor_fn calls every agent's
        # sample_actions with temperature= uniformly (it scales IQL/ACIQL's Gaussian actor std),
        # but CGQ's actor is a deterministic flow/distillation actor with no equivalent knob.
        del temperature
        if self.config["actor_type"] == "distill-ddpg":
            noises = jax.random.normal(
                rng,
                (
                    *observations.shape[: -len(self.config['ob_dims'])],
                    self.config['action_dim']
                ),
            )
            actions = self.network.select(f'step_actor_onestep_flow')(observations, noises)
            actions = jnp.clip(actions, -1, 1)

        elif self.config["actor_type"] == "best-of-n":
            noises = jax.random.normal(
                rng,
                (
                    *observations.shape[: -len(self.config['ob_dims'])],
                    self.config["actor_num_samples"], self.config['action_dim']
                )
            )
            observations = jnp.repeat(observations[..., None, :], self.config["actor_num_samples"], axis=-2)
            actions = self.compute_step_flow_actions(observations, noises)
            actions = jnp.clip(actions, -1, 1)
            if self.config["q_agg"] == "mean":
                q = self.network.select("step_critic")(observations, actions).mean(axis=0)
            else:
                q = self.network.select("step_critic")(observations, actions).min(axis=0)
            indices = jnp.argmax(q, axis=-1)

            bshape = indices.shape
            indices = indices.reshape(-1)
            bsize = len(indices)
            actions = jnp.reshape(actions, (-1, self.config["actor_num_samples"], self.config['action_dim']))[jnp.arange(bsize), indices, :].reshape(
                bshape + (self.config['action_dim'],))
        return actions

    @jax.jit
    def compute_step_flow_actions(
        self,
        observations,
        noises,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        if self.config['encoder'] is not None:
            observations = self.network.select('step_actor_bc_flow_encoder')(observations)
        actions = noises # (B, action_dim)
        # Euler method.
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps']) # (B, 1), Last dim is for time.
            vels = self.network.select('step_actor_bc_flow')(observations, actions, t, is_encoded=True)
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
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]
        chunk_actions = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
        chunk_action_dim = chunk_actions.shape[-1] # action_dim * horizon_length

        ex_step_actions = ex_actions
        ex_chunk_actions = chunk_actions


        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['chunk_critic'] = encoder_module()
            encoders['chunk_actor_bc_flow'] = encoder_module()
            encoders['chunk_actor_onestep_flow'] = encoder_module()
            encoders['step_critic'] = encoder_module()
            encoders['step_actor_bc_flow'] = encoder_module()
            encoders['step_actor_onestep_flow'] = encoder_module()

        # Define networks.
        chunk_critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_qs'],
            encoder=encoders.get('chunk_critic'),
        )

        chunk_actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=chunk_action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('chunk_actor_bc_flow'),
            use_fourier_features=config["use_fourier_features"],
            fourier_feature_dim=config["fourier_feature_dim"],
        )
        chunk_actor_onestep_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=chunk_action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('chunk_actor_onestep_flow'),
        )

        # Define networks.
        step_critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_qs'],
            encoder=encoders.get('step_critic'),
        )

        step_actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('step_actor_bc_flow'),
            use_fourier_features=config["use_fourier_features"],
            fourier_feature_dim=config["fourier_feature_dim"],
        )
        step_actor_onestep_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('step_actor_onestep_flow'),
        )


        network_info = dict(
            chunk_actor_bc_flow=(chunk_actor_bc_flow_def, (ex_observations, ex_chunk_actions, ex_times)),
            chunk_actor_onestep_flow=(chunk_actor_onestep_flow_def, (ex_observations, ex_chunk_actions)),
            chunk_critic=(chunk_critic_def, (ex_observations, ex_chunk_actions)),
            target_chunk_critic=(copy.deepcopy(chunk_critic_def), (ex_observations, ex_chunk_actions)),
            step_actor_bc_flow=(step_actor_bc_flow_def, (ex_observations, ex_step_actions, ex_times)),
            step_actor_onestep_flow=(step_actor_onestep_flow_def, (ex_observations, ex_step_actions)),
            step_critic=(step_critic_def, (ex_observations, ex_step_actions)),
            target_step_critic=(copy.deepcopy(step_critic_def), (ex_observations, ex_step_actions)),

        )
        if encoders.get('actor_bc_flow') is not None:
            network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (ex_observations,))
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        if config["weight_decay"] > 0.:
            network_tx = optax.adamw(learning_rate=config['lr'], weight_decay=config["weight_decay"])
        else:
            network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params

        params[f'modules_target_step_critic'] = params[f'modules_step_critic']
        params[f'modules_target_chunk_critic'] = params[f'modules_chunk_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        config['chunk_action_dim'] = chunk_action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():

    config = ml_collections.ConfigDict(
        dict(
            agent_name='cgq_gripcond',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=True,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            q_agg='mean',  # Aggregation method for target Q values.
            step_alpha=100.0, # BC coefficient (need to be tuned for each environment).
            alpha=100.0,  # BC coefficient (need to be tuned for each environment).
            num_qs=2, # critic ensemble size
            flow_steps=10,  # Number of flow steps.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            horizon_length=ml_collections.config_dict.placeholder(int), # will be set
            action_chunking=True,  # False means n-step return
            actor_type="distill-ddpg",
            chunk_actor_type="distill-ddpg",
            use_fourier_features=False,
            fourier_feature_dim=64,
            weight_decay=0.,
            anchor_expectile=0.95,
            step_critic_type="actor-critic",  # "SARSA" or "actor-critic"
            # beta (single scalar in cgq.py) is replaced by a gripper-conditional pair -- see
            # module docstring.
            beta_reaching=1.0,   # anchor weight while NOT gripping (large -- trust chunk_critic)
            beta_gripping=0.1,   # anchor weight while gripping (small -- trust chunk_critic less)
            td_loss=1.0,  # td loss coefficient
            # Gripper-conditional anchor-weight switch (cube-family envs only -- see
            # dqc_gripcond.py's module docstring for the manipspace_env.py observation-layout
            # provenance of these indices/defaults).
            gripper_opening_obs_idx=17,
            gripper_opening_scaler=3.0,
            gripper_closed_threshold=0.5,
        )
    )
    return config
