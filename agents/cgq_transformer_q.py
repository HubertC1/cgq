import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, CausalChunkQTransformer, ensemblize


class CGQTransformerQAgent(flax.struct.PyTreeNode):
    """CGQ (agents/cgq.py), with its two independent MLP critics (chunk_critic, step_critic)
    replaced by a single ensembled CausalChunkQTransformer with chunk_sizes=(H, 1) -- one shared
    forward pass yields both the H-step and 1-step head at once, ensembled the same way as CGQ's
    own critics (config['num_qs'] independent copies via utils.networks.ensemblize, aggregated via
    config['q_agg']). This is the one deliberate architecture change; everything else -- the DDPG-
    style backup through actor-proposed actions (no V network, no expectile-max, in contrast to
    agents/curriculum_transformer_q.py's IQL-style approach), the anchor loss pulling the 1-step
    head toward the H-step head, the dual BC-flow + one-step-distilled actor pairs, all loss
    weights and mode toggles -- is transplanted from cgq.py as-is, including its exact defaults.

    Because both heads now share one network, the anchor loss's ac_values (CGQ's stopgrad'd
    Q_H(s,a_{1:H}), used as the anchor target) and chunk_critic_loss's own live Q_H(s,a_{1:H})
    read off the *same* underlying quantity -- computed once per step (with params=grad_params, so
    gradient flows for chunk_critic_loss's own term), with jax.lax.stop_gradient applied only where
    forming ac_values, rather than a second forward pass with params withheld (CGQ's own
    mechanism, since chunk_critic and step_critic were separate networks there -- withholding
    params from a *different* network naturally blocks gradient; here it's the same network called
    once, so the block has to be explicit).

    Trains its own actors jointly (like cgq.py, dqc.py, acfql.py) rather than reusing the shared
    frozen-BC-flow convention agents/curriculum_qchunk.py and agents/curriculum_transformer_q.py
    use -- CGQ is already one of this repo's "joint actor-critic, exempt from fixed policy
    extraction" baselines (see main.py's merge-logic comment), and this agent stays in that family.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff ** 2)

    def _pad_to_horizon(self, actions):
        """(..., action_dim) single-step actions -> (..., H, action_dim), zero-padded after
        position 0. Positions after 0 are causally invisible to the h=1 head (CausalChunkQTransformer
        by construction), so this is exactly equivalent to CGQ's step_critic taking a bare
        single-step action -- just re-packaged into the shape the shared transformer needs."""
        horizon = self.config['horizon_length']
        zeros_tail = jnp.zeros(actions.shape[:-1] + (horizon - 1, actions.shape[-1]))
        return jnp.concatenate([actions[..., None, :], zeros_tail], axis=-2)

    def _agg(self, qs):
        return qs.min(axis=0) if self.config['q_agg'] == 'min' else qs.mean(axis=0)

    def chunk_critic_loss(self, batch, grad_params, q_all_live, rng):
        """DDPG-style TD loss for the H-step head: bootstraps off target_transformer's H-head at
        the next state, evaluated on an action sampled from chunk_actor_onestep_flow (or
        best-of-n against chunk_actor_bc_flow) -- not an expectile-regressed V(s)."""
        horizon = self.config['horizon_length']
        action_dim = self.config['action_dim']

        rng, sample_rng = jax.random.split(rng)
        next_obs = batch['next_observations'][..., -1, :]
        next_actions_flat = self.sample_chunk_actions(next_obs, rng=sample_rng)  # (..., H*action_dim)
        next_actions_chunk = next_actions_flat.reshape(*next_actions_flat.shape[:-1], horizon, action_dim)

        next_qs = self.network.select('target_transformer')(next_obs, next_actions_chunk)[horizon]
        next_q = self._agg(next_qs)

        target_q = batch['rewards'][..., -1] + (self.config['discount'] ** horizon) * batch['masks'][..., -1] * next_q

        q = q_all_live[horizon]  # (num_qs, B) -- live, gradient flows (q_all_live was computed with params=grad_params)
        critic_loss = (jnp.square(q - target_q[None]) * batch['valid'][..., -1][None]).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def chunk_actor_loss(self, batch, grad_params, rng):
        """BC flow + (distill-ddpg) distillation + Q-loss climbing the H-step head."""
        horizon = self.config['horizon_length']
        action_dim = self.config['action_dim']
        chunk_action_dim = horizon * action_dim

        batch_actions = jnp.reshape(batch['actions'], (batch['actions'].shape[0], -1))
        batch_size = batch_actions.shape[0]
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        x_0 = jax.random.normal(x_rng, (batch_size, chunk_action_dim))
        x_1 = batch_actions
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('chunk_actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)
        bc_flow_loss = jnp.mean(
            jnp.reshape((pred - vel) ** 2, (batch_size, horizon, action_dim)) * batch['valid'][..., None]
        )

        if self.config['chunk_actor_type'] == 'distill-ddpg':
            rng, noise_rng = jax.random.split(rng)
            noises = jax.random.normal(noise_rng, (batch_size, chunk_action_dim))
            target_flow_actions = self.compute_chunk_flow_actions(batch['observations'], noises=noises)
            actor_actions = self.network.select('chunk_actor_onestep_flow')(batch['observations'], noises, params=grad_params)
            distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

            actor_actions = jnp.clip(actor_actions, -1, 1)
            actor_actions_chunk = actor_actions.reshape(batch_size, horizon, action_dim)
            # No params=grad_params here -- gradient must flow into the actor (via actor_actions,
            # which was computed with grad_params) but not into the critic itself from this term.
            qs = self.network.select('transformer')(batch['observations'], actor_actions_chunk)[horizon]
            q = jnp.mean(qs, axis=0)  # CGQ hardcodes mean for the actor's own Q-loss regardless of q_agg.
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

    def step_critic_loss(self, batch, grad_params, q_all_live, rng):
        """SHARSA-style TD loss for the 1-step head, plus the anchor loss pulling it toward the
        (stopgrad'd) H-step head's value -- CGQ's mechanism for letting the more-reliable chunk
        estimate teach the 1-step one, expectile or td-inversion (config['anchor_loss_type'])."""
        horizon = self.config['horizon_length']
        rng, sample_rng = jax.random.split(rng)
        next_obs = batch['next_observations'][..., 0, :]
        batch_rewards = batch['rewards'][..., 0]
        batch_masks = batch['masks'][..., 0]

        if self.config['step_critic_type'] == 'SARSA':
            next_actions = batch['actions'][..., 1, :]
        elif self.config['step_critic_type'] == 'actor-critic':
            next_actions = self.sample_actions(next_obs, rng=sample_rng)
        else:
            raise NotImplementedError

        next_qs = self.network.select('target_transformer')(next_obs, self._pad_to_horizon(next_actions))[1]
        next_q = self._agg(next_qs)
        target_q = batch_rewards + self.config['discount'] * batch_masks * next_q

        q = q_all_live[1]  # (num_qs, B) -- live, Q_1(s, a_1) at the dataset's real first action.
        td_loss = jnp.mean(jnp.square(q - target_q))  # matches cgq.py exactly -- no valid-masking here.

        ac_values = jax.lax.stop_gradient(q_all_live[horizon])  # CGQ's ac_values: same network, no
        # gradient into the H-head from the anchor term (there it's a no-params-passed call to a
        # *different* network; here it's the same network, so the block has to be explicit).

        if self.config['anchor_loss_type'] == 'td':
            anchor_q = self.network.select('transformer')(
                batch['next_observations'][..., -1, :],
                self._pad_to_horizon(batch['actions'][..., -1, :]),
                params=grad_params,
            )[1]
            anchor_target = (ac_values - batch['rewards'][..., -1]) / (self.config['discount'] ** horizon)
            anchor_loss = (jnp.square(anchor_q - anchor_target) * batch['valid'][..., -1][None]).mean()
        else:
            diff = ac_values - q
            anchor_loss = (self.expectile_loss(diff, diff, self.config['anchor_expectile']) * batch['valid'][..., -1][None]).mean()

        critic_loss = self.config['td_loss'] * td_loss + self.config['beta'] * anchor_loss

        return critic_loss, {
            'step_critic_loss': critic_loss,
            'td_loss': td_loss,
            'anchor_loss': anchor_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def step_actor_loss(self, batch, grad_params, rng):
        """BC flow + (distill-ddpg) distillation + Q-loss climbing the 1-step head."""
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

        if self.config['actor_type'] == 'distill-ddpg':
            rng, noise_rng = jax.random.split(rng)
            noises = jax.random.normal(noise_rng, (batch_size, action_dim))
            target_flow_actions = self.compute_step_flow_actions(batch['observations'], noises=noises)
            actor_actions = self.network.select('step_actor_onestep_flow')(batch['observations'], noises, params=grad_params)
            distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

            actor_actions = jnp.clip(actor_actions, -1, 1)
            qs = self.network.select('transformer')(batch['observations'], self._pad_to_horizon(actor_actions))[1]
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

        # One shared live pass on the real (s_t, a_{1:H}) chunk -- gives both heads at once, reused
        # by chunk_critic_loss (its own H-head prediction) and step_critic_loss (its own 1-head
        # prediction, and the anchor's ac_values = stopgrad(this same H-head output)).
        q_all_live = self.network.select('transformer')(batch['observations'], batch['actions'], params=grad_params)

        chunk_critic_loss, chunk_critic_info = self.chunk_critic_loss(batch, grad_params, q_all_live, chunk_critic_rng)
        for k, v in chunk_critic_info.items():
            info[f'chunk_critic/{k}'] = v

        chunk_actor_loss, chunk_actor_info = self.chunk_actor_loss(batch, grad_params, chunk_actor_rng)
        for k, v in chunk_actor_info.items():
            info[f'chunk_actor/{k}'] = v

        step_critic_loss, step_critic_info = self.step_critic_loss(batch, grad_params, q_all_live, step_critic_rng)
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
        agent.target_update(new_network, 'transformer')
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
    def sample_chunk_actions(self, observations, rng=None):
        horizon = self.config['horizon_length']
        action_dim = self.config['action_dim']
        chunk_action_dim = horizon * action_dim

        if self.config['chunk_actor_type'] == 'distill-ddpg':
            noises = jax.random.normal(rng, (*observations.shape[: -len(self.config['ob_dims'])], chunk_action_dim))
            actions = self.network.select('chunk_actor_onestep_flow')(observations, noises)
            actions = jnp.clip(actions, -1, 1)

        elif self.config['chunk_actor_type'] == 'best-of-n':
            noises = jax.random.normal(
                rng,
                (*observations.shape[: -len(self.config['ob_dims'])], self.config['chunk_actor_num_samples'], chunk_action_dim),
            )
            obs_tiled = jnp.repeat(observations[..., None, :], self.config['chunk_actor_num_samples'], axis=-2)
            actions = self.compute_chunk_flow_actions(obs_tiled, noises)
            actions = jnp.clip(actions, -1, 1)

            actions_chunk = actions.reshape(*actions.shape[:-1], horizon, action_dim)
            qs = self.network.select('transformer')(obs_tiled, actions_chunk)[horizon]
            q = self._agg(qs)
            indices = jnp.argmax(q, axis=-1)

            bshape = indices.shape
            indices = indices.reshape(-1)
            bsize = len(indices)
            actions = jnp.reshape(actions, (-1, self.config['chunk_actor_num_samples'], chunk_action_dim))[jnp.arange(bsize), indices, :].reshape(
                bshape + (chunk_action_dim,))

        return actions

    @jax.jit
    def compute_chunk_flow_actions(self, observations, noises):
        """Compute actions from the BC flow model using the Euler method."""
        actions = noises
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('chunk_actor_bc_flow')(observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
        return actions

    @jax.jit
    def sample_actions(self, observations, rng=None, temperature=1.0):
        # temperature accepted but unused, matching every other flow-policy agent in this repo.
        del temperature
        if self.config['actor_type'] == 'distill-ddpg':
            noises = jax.random.normal(rng, (*observations.shape[: -len(self.config['ob_dims'])], self.config['action_dim']))
            actions = self.network.select('step_actor_onestep_flow')(observations, noises)
            actions = jnp.clip(actions, -1, 1)

        elif self.config['actor_type'] == 'best-of-n':
            noises = jax.random.normal(
                rng,
                (*observations.shape[: -len(self.config['ob_dims'])], self.config['actor_num_samples'], self.config['action_dim']),
            )
            obs_tiled = jnp.repeat(observations[..., None, :], self.config['actor_num_samples'], axis=-2)
            actions = self.compute_step_flow_actions(obs_tiled, noises)
            actions = jnp.clip(actions, -1, 1)

            qs = self.network.select('transformer')(obs_tiled, self._pad_to_horizon(actions))[1]
            q = self._agg(qs)
            indices = jnp.argmax(q, axis=-1)

            bshape = indices.shape
            indices = indices.reshape(-1)
            bsize = len(indices)
            actions = jnp.reshape(actions, (-1, self.config['actor_num_samples'], self.config['action_dim']))[jnp.arange(bsize), indices, :].reshape(
                bshape + (self.config['action_dim'],))
        return actions

    @jax.jit
    def compute_step_flow_actions(self, observations, noises):
        """Compute actions from the BC flow model using the Euler method."""
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
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example observations.
            ex_actions: Example actions (single-step).
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        horizon = config['horizon_length']
        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]
        chunk_action_dim = horizon * action_dim

        ex_chunk_actions_flat = jnp.concatenate([ex_actions] * horizon, axis=-1)  # (chunk_action_dim,) -- for the actor flows.
        ex_chunk_actions = jnp.stack([ex_actions] * horizon, axis=0)  # (horizon, action_dim) -- for the transformer.

        EnsembledTransformer = ensemblize(CausalChunkQTransformer, config['num_qs'])
        transformer_def = EnsembledTransformer(
            chunk_sizes=(horizon, 1),
            hidden_dim=config['transformer_hidden_dim'],
            num_layers=config['transformer_num_layers'],
            num_heads=config['transformer_num_heads'],
            mlp_ratio=config['transformer_mlp_ratio'],
            layer_norm=config['transformer_layer_norm'],
        )

        chunk_actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=chunk_action_dim,
            layer_norm=config['actor_layer_norm'],
            use_fourier_features=config['use_fourier_features'],
            fourier_feature_dim=config['fourier_feature_dim'],
        )
        chunk_actor_onestep_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=chunk_action_dim,
            layer_norm=config['actor_layer_norm'],
        )
        step_actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            use_fourier_features=config['use_fourier_features'],
            fourier_feature_dim=config['fourier_feature_dim'],
        )
        step_actor_onestep_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
        )

        network_info = dict(
            transformer=(transformer_def, (ex_observations, ex_chunk_actions)),
            target_transformer=(copy.deepcopy(transformer_def), (ex_observations, ex_chunk_actions)),
            chunk_actor_bc_flow=(chunk_actor_bc_flow_def, (ex_observations, ex_chunk_actions_flat, ex_times)),
            chunk_actor_onestep_flow=(chunk_actor_onestep_flow_def, (ex_observations, ex_chunk_actions_flat)),
            step_actor_bc_flow=(step_actor_bc_flow_def, (ex_observations, ex_actions, ex_times)),
            step_actor_onestep_flow=(step_actor_onestep_flow_def, (ex_observations, ex_actions)),
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
        params['modules_target_transformer'] = params['modules_transformer']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        config['chunk_action_dim'] = chunk_action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='cgq_transformer_q',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            layer_norm=True,  # Unused (kept for main.py registry-key compatibility) -- the
            # transformer's own layer_norm is transformer_layer_norm below.
            actor_layer_norm=True,
            discount=0.99,
            tau=0.005,  # Target network update rate.
            q_agg='mean',  # Aggregation method for target Q values -- CGQ's own default; note this
            # repo's IQL-family antmaze-large/giant-navigate configs all use 'min' instead, after
            # documented mean-of-ensemble instability there. Kept at CGQ's default here as asked.
            step_alpha=100.0,  # BC coefficient for the step (1-step) actor.
            alpha=100.0,  # BC coefficient for the chunk (H-step) actor.
            num_qs=2,  # Critic ensemble size -- now applied to the shared transformer via
            # utils.networks.ensemblize (full independent copies, not a shared-trunk multi-head).
            flow_steps=10,  # Number of flow steps.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            horizon_length=ml_collections.config_dict.placeholder(int),  # H -- the chunk_sizes=(H,1)
            # transformer's max chunk length; drives main.py's dataset.sample_sequence length.
            actor_type='distill-ddpg',  # 'distill-ddpg' | 'best-of-n' -- step (1-step) actor.
            chunk_actor_type='distill-ddpg',  # 'distill-ddpg' | 'best-of-n' -- chunk (H-step) actor.
            use_fourier_features=False,
            fourier_feature_dim=64,
            weight_decay=0.,
            anchor_loss_type='expectile',  # 'expectile' (regress Q_1(s,a_1) toward the H-head) or
            # 'td' (TD-guided: regress Q_1(s',a_H) toward the implied bootstrap value) -- CGQ's own default.
            anchor_expectile=0.95,
            step_critic_type='actor-critic',  # 'SARSA' | 'actor-critic'.
            beta=0.01,  # Anchor loss coefficient.
            td_loss=1.0,  # TD loss coefficient.
            chunk_actor_num_samples=32,  # N candidates for chunk actor best-of-n (chunk_actor_type only).
            actor_num_samples=32,  # N candidates for step actor best-of-n (actor_type only).

            # Transformer critic architecture (the one deliberate architecture change from cgq.py).
            transformer_hidden_dim=128,
            transformer_num_layers=2,
            transformer_num_heads=8,
            transformer_mlp_ratio=4,
            transformer_layer_norm=True,
        )
    )
    return config
