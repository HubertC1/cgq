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


class TDCGQAgent(flax.struct.PyTreeNode):
    """CGQ (agents/cgq.py) with two changes to the one-step critic, everything else identical:

      1. The anchor/regularization loss is gone. cgq.py's step_critic_loss pulls Q_1 toward the
         (stopgrad'd) chunk critic via an explicit extra term (expectile or td-inversion,
         config['beta']-weighted) on top of its own TD loss. Here step_critic_loss is just the TD
         loss -- no anchor term, no config['anchor_loss_type']/['anchor_expectile']/['beta'] (removed
         from get_config() entirely, not just unused).

      2. The one-step TD target's bootstrap is no longer purely Q(s', pi) (the step critic's own
         next-state value under its own policy). It's a weighted blend with the chunk critic's value
         at that *same* next state, under the chunk actor's own proposed chunk:

             next_q = w(t) * Q(s', pi(s')) + (1 - w(t)) * Q_c(s', pi_c(s'))
             target_q = r_1 + discount * mask_1 * next_q

         w(t) (config['weight_schedule']):
           - 'linear': w(t) = clip(step / anneal_steps, 0, 1) -- starts at 0 (100% chunk critic)
             and reaches 1 (100% step critic, i.e. cgq.py's original target) at step=anneal_steps.
           - 'constant': w(t) = config['constant_step_weight'] always (default 0.5, i.e. a fixed
             1:1 blend the whole run).

      This is why a step counter had to be added -- cgq.py's agent state (rng, network, config) has
      no notion of training step, since nothing in it needed one.
      """

    rng: Any
    step: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff ** 2)

    def _step_weight(self):
        """w(t): weight on the step critic's own Q(s',pi) in the one-step TD target's bootstrap
        blend. (1 - w(t)) is the chunk critic Q_c(s',pi_c)'s weight."""
        if self.config['weight_schedule'] == 'constant':
            return jnp.asarray(self.config['constant_step_weight'], dtype=jnp.float32)
        anneal_steps = max(self.config['anneal_steps'], 1)
        return jnp.clip(self.step.astype(jnp.float32) / anneal_steps, 0.0, 1.0)

    def chunk_critic_loss(self, batch, grad_params, rng):
        """Unchanged from cgq.py."""

        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]

        rng, sample_rng = jax.random.split(rng)
        next_actions = self.sample_chunk_actions(batch['next_observations'][..., -1, :], rng=sample_rng)

        next_qs = self.network.select('target_chunk_critic')(batch['next_observations'][..., -1, :], actions=next_actions)
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
        """Unchanged from cgq.py."""
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]
        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch_actions
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('chunk_actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)

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
            rng, noise_rng = jax.random.split(rng)
            noises = jax.random.normal(noise_rng, (batch_size, action_dim))
            target_flow_actions = self.compute_chunk_flow_actions(batch['observations'], noises=noises)
            actor_actions = self.network.select('chunk_actor_onestep_flow')(batch['observations'], noises, params=grad_params)
            distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

            actor_actions = jnp.clip(actor_actions, -1, 1)

            qs = self.network.select('chunk_critic')(batch['observations'], actions=actor_actions)
            q = jnp.mean(qs, axis=0)
            q_loss = -q.mean()

            if self.config['normalize_q_loss']:
                lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
                q_loss = lam * q_loss
        else:
            distill_loss = jnp.zeros(())
            q_loss = jnp.zeros(())

        actor_loss = bc_flow_loss + self.config['alpha'] * distill_loss + q_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'distill_loss': distill_loss,
            'q_loss': q_loss,
        }

    def step_critic_loss(self, batch, grad_params, rng):
        """One-step TD loss only -- no anchor/regularization term. The bootstrap blends the step
        critic's own Q(s',pi) with the chunk critic's Q_c(s',pi_c) at the same next state, weighted
        by self._step_weight()."""
        rng, sample_rng, chunk_sample_rng = jax.random.split(rng, 3)
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
        next_qs = self.network.select('target_step_critic')(next_obs, actions=next_actions)
        if self.config['q_agg'] == 'min':
            next_q_step = next_qs.min(axis=0)
        else:
            next_q_step = next_qs.mean(axis=0)

        # Q_c(s', pi_c(s')) -- the chunk critic evaluated at the SAME next state (after 1 step),
        # under a fresh chunk proposed by the chunk actor starting there. Target network, matching
        # this repo's convention for whatever feeds a bootstrap target.
        next_chunk_actions = self.sample_chunk_actions(next_obs, rng=chunk_sample_rng)
        next_chunk_qs = self.network.select('target_chunk_critic')(next_obs, actions=next_chunk_actions)
        if self.config['q_agg'] == 'min':
            next_q_chunk = next_chunk_qs.min(axis=0)
        else:
            next_q_chunk = next_chunk_qs.mean(axis=0)

        w = self._step_weight()
        next_q = w * next_q_step + (1 - w) * next_q_chunk

        target_q = batch_rewards + self.config['discount'] * batch_masks * next_q

        q = self.network.select('step_critic')(batch['observations'], actions=actions, params=grad_params)
        td_loss = jnp.mean(jnp.square(q - target_q))

        critic_loss = self.config['td_loss'] * td_loss

        return critic_loss, {
            'step_critic_loss': critic_loss,
            'td_loss': td_loss,
            'weight_step': w,
            'next_q_step_mean': next_q_step.mean(),
            'next_q_chunk_mean': next_q_chunk.mean(),
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def step_actor_loss(self, batch, grad_params, rng):
        """Unchanged from cgq.py."""
        batch_actions = batch['actions'][..., 0, :]
        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch_actions
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('step_actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)

        bc_flow_loss = jnp.mean((pred - vel) ** 2)

        if self.config["actor_type"] == "distill-ddpg":
            rng, noise_rng = jax.random.split(rng)
            noises = jax.random.normal(noise_rng, (batch_size, action_dim))
            target_flow_actions = self.compute_step_flow_actions(batch['observations'], noises=noises)
            actor_actions = self.network.select('step_actor_onestep_flow')(batch['observations'], noises, params=grad_params)
            distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

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
        return agent.replace(network=new_network, rng=new_rng, step=agent.step + 1), info

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
        """Unchanged from cgq.py."""
        if self.config["chunk_actor_type"] == "distill-ddpg":
            noises = jax.random.normal(
                rng,
                (
                    *observations.shape[: -len(self.config['ob_dims'])],
                    self.config['action_dim'] *
                        (self.config['horizon_length'] if self.config["action_chunking"] else 1),
                ),
            )
            actions = self.network.select('chunk_actor_onestep_flow')(observations, noises)
            actions = jnp.clip(actions, -1, 1)

        elif self.config["chunk_actor_type"] == "best-of-n":
            action_dim = self.config['action_dim'] * \
                        (self.config['horizon_length'] if self.config["action_chunking"] else 1)
            noises = jax.random.normal(
                rng,
                (
                    *observations.shape[: -len(self.config['ob_dims'])],
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
        """Unchanged from cgq.py."""
        if self.config['encoder'] is not None:
            observations = self.network.select('chunk_actor_bc_flow_encoder')(observations)
        actions = noises
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
        """Unchanged from cgq.py."""
        del temperature
        if self.config["actor_type"] == "distill-ddpg":
            noises = jax.random.normal(
                rng,
                (
                    *observations.shape[: -len(self.config['ob_dims'])],
                    self.config['action_dim']
                ),
            )
            actions = self.network.select('step_actor_onestep_flow')(observations, noises)
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
        """Unchanged from cgq.py."""
        if self.config['encoder'] is not None:
            observations = self.network.select('step_actor_bc_flow_encoder')(observations)
        actions = noises
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
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
        """Create a new agent. Identical to cgq.py's create() except for the added `step` field.

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
        chunk_action_dim = chunk_actions.shape[-1]

        ex_step_actions = ex_actions
        ex_chunk_actions = chunk_actions

        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['chunk_critic'] = encoder_module()
            encoders['chunk_actor_bc_flow'] = encoder_module()
            encoders['chunk_actor_onestep_flow'] = encoder_module()
            encoders['step_critic'] = encoder_module()
            encoders['step_actor_bc_flow'] = encoder_module()
            encoders['step_actor_onestep_flow'] = encoder_module()

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

        params['modules_target_step_critic'] = params['modules_step_critic']
        params['modules_target_chunk_critic'] = params['modules_chunk_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        config['chunk_action_dim'] = chunk_action_dim

        return cls(rng, step=jnp.zeros((), dtype=jnp.int32), network=network, config=flax.core.FrozenDict(**config))


def get_config():

    config = ml_collections.ConfigDict(
        dict(
            agent_name='tdcgq',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            actor_layer_norm=True,
            discount=0.99,
            tau=0.005,
            q_agg='mean',
            step_alpha=100.0,
            alpha=100.0,
            num_qs=2,
            flow_steps=10,
            normalize_q_loss=False,
            encoder=ml_collections.config_dict.placeholder(str),
            horizon_length=ml_collections.config_dict.placeholder(int),
            action_chunking=True,
            actor_type="distill-ddpg",
            chunk_actor_type="distill-ddpg",
            use_fourier_features=False,
            fourier_feature_dim=64,
            weight_decay=0.,
            step_critic_type="actor-critic",  # "SARSA" or "actor-critic"
            td_loss=1.0,  # td loss coefficient -- now the step critic's only loss term.
            chunk_actor_num_samples=32,
            actor_num_samples=32,

            # The one-step TD target's bootstrap blend (replaces cgq.py's anchor/reg loss entirely
            # -- anchor_loss_type/anchor_expectile/beta are gone, not just unused).
            weight_schedule='linear',  # 'linear' | 'constant'.
            anneal_steps=1_000_000,  # 'linear' only: w(t)=clip(step/anneal_steps,0,1) reaches 1
            # (100% step critic, cgq.py's original target) at step=anneal_steps; starts at 0 (100%
            # chunk critic) at step=0.
            constant_step_weight=0.5,  # 'constant' only: fixed weight on the step critic, the whole run.
        )
    )
    return config
