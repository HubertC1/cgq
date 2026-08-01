import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Actor, ActorVectorField, Value
from utils.flow_policy import flow_rejection_sample_actions, load_frozen_bc_flow_params


class ACIQLAgent(flax.struct.PyTreeNode):
    """Action-chunked IQL (ACIQL).

    Same n-step-return idea as iql.py's horizon_length>1 case, except the n-step bootstrap is
    assigned to a *chunked* critic Q(s, a_chunk) instead of Q(s, a) -- a_chunk is the full
    horizon_length-action sequence, flattened, matching acfql.py's chunked-critic convention. V(s)
    is expectile-regressed against Q(s, a_chunk) using the dataset's actual chunk (never a
    policy-sampled one), and the AWR actor is a chunk policy trained on the same dataset chunk,
    weighted by exp(alpha * (Q(s, a_chunk) - V(s))).

    Every loss below only ever evaluates Q at a *logged* a_chunk -- nowhere does anything
    bootstrap through an action selected by a still-training policy, so this keeps IQL's in-sample
    guarantee (immune to suboptimal-data / OOD-action extrapolation) intact even though the domain
    of Q is now the full chunk rather than a single action. That's a different property from
    stochastic-dynamics immunity, though: committing to horizon_length actions open-loop still
    means this can't react to what actually happens mid-chunk, so it inherits the same
    OLC-violation cost that grows with h as DQC/CGQ/QC-FQL (CLAUDE.md's chunked-critic vocabulary)
    -- this agent isolates whether chunking alone (with a safe bootstrap) gets the claimed
    horizon-reduction benefit, not a claim to the empty vertex of the three-way tradeoff.

    Policy extraction (config['policy_method']) is independent of the above and defaults to the
    AWR chunk policy described above for backward compatibility. Setting policy_method=
    'flow_rejection' swaps in CLAUDE.md's Phase 1 target extraction instead: config['bc_checkpoint']
    must point at an agents/bc.py flow-BC checkpoint (policy_method='flow') pretrained once,
    externally, on this same dataset with horizon_length matching this agent's own -- see that
    module's class docstring for why it must be pretrained and frozen rather than trained jointly
    here (a shared eval-time policy across baselines is the whole point of CLAUDE.md's Phase 1
    methodology; an independently-SGD'd copy per critic run would reintroduce exactly the confound
    that's meant to eliminate). This agent never trains that policy -- it's loaded once in create()
    and never receives a gradient afterward (same zero-gradient mechanism already used for
    target_critic, just without even a polyak update), then used for best-of-N rejection sampling
    over the flattened chunk against this agent's own (unmodified) critic; see utils/flow_policy.py.
    Either way, value_loss/critic_loss above are completely unaffected -- only how a chunk gets
    proposed from the learned Q changes.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def value_loss(self, batch, grad_params):
        """V(s) expectile-regressed against target_critic(s, a_chunk) using the dataset's chunk."""
        chunk_actions = jnp.reshape(batch['actions'], (batch['actions'].shape[0], -1))
        qs = self.network.select('target_critic')(batch['observations'], actions=chunk_actions)
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
        """Q(s, a_chunk) TD-regressed against the h-step return + discount^h * V(s_h)."""
        next_v = self.network.select('value')(batch['next_observations'][..., -1, :])
        target_q = batch['rewards'][..., -1] + (
            self.config['discount'] ** self.config['horizon_length']
        ) * batch['masks'][..., -1] * next_v
        chunk_actions = jnp.reshape(batch['actions'], (batch['actions'].shape[0], -1))
        qs = self.network.select('critic')(batch['observations'], actions=chunk_actions, params=grad_params)
        critic_loss = (jnp.square(qs - target_q[None]) * batch['valid'][..., -1][None]).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': qs.mean(),
            'q_max': qs.max(),
            'q_min': qs.min(),
        }

    def actor_loss(self, batch, grad_params):
        """AWR over the chunk policy: BC log-prob on the dataset's chunk, weighted by
        exp(alpha * advantage). Same alpha-direction caveat as iql.py: higher alpha = more greedy
        toward Q, opposite direction from FQL's BC-strength alpha.
        """
        chunk_actions = jnp.reshape(batch['actions'], (batch['actions'].shape[0], -1))
        v = self.network.select('value')(batch['observations'])
        qs = self.network.select('target_critic')(batch['observations'], actions=chunk_actions)
        q = jnp.min(qs, axis=0) if self.config['q_agg'] == 'min' else jnp.mean(qs, axis=0)
        adv = q - v

        exp_a = jnp.exp(adv * self.config['alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        dist = self.network.select('actor')(batch['observations'], params=grad_params)
        log_prob = dist.log_prob(chunk_actions)

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

        loss = value_loss + critic_loss
        if self.config['policy_method'] == 'awr':
            # 'flow_rejection' has nothing to train here -- config['bc_checkpoint'] is loaded once
            # in create() and frozen (see class docstring), so there's no actor loss term at all in
            # that mode, not just a different one.
            actor_loss, actor_info = self.actor_loss(batch, grad_params)
            for k, v in actor_info.items():
                info[f'actor/{k}'] = v
            loss = loss + actor_loss

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
        """Samples a full action chunk, flattened -- matches acfql.py's output convention, so the
        existing eval loop's chunk-unpacking (reshape(-1, action_dim) into an action queue) works
        unchanged.
        """
        if self.config['policy_method'] == 'flow_rejection':
            full_action_dim = self.config['action_dim'] * self.config['horizon_length']
            actor_bc_flow_fn = lambda o, x, t: self.network.select('actor_bc_flow')(o, x, t)
            critic_fn = lambda o, a: self.network.select('critic')(o, actions=a)
            return flow_rejection_sample_actions(
                actor_bc_flow_fn, critic_fn, observations, rng,
                num_samples=self.config['actor_num_samples'],
                flow_steps=self.config['flow_steps'],
                action_dim=full_action_dim,
                q_agg=self.config['q_agg'],
            )
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
            ex_actions: Example batch of actions (single-step; will be tiled into a chunk).
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
            encoders['value'] = encoder_module()
            encoders['critic'] = encoder_module()
            if config['policy_method'] == 'flow_rejection':
                encoders['actor_bc_flow'] = encoder_module()
            else:
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

        network_info = dict(
            value=(value_def, (ex_observations,)),
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
        )

        if config['policy_method'] == 'flow_rejection':
            assert config['bc_checkpoint'] is not None, (
                "policy_method='flow_rejection' requires config['bc_checkpoint'] -- a pretrained "
                "agents/bc.py flow-BC checkpoint path. This agent never trains that policy itself."
            )
            assert config['weight_decay'] == 0., (
                "policy_method='flow_rejection' loads a frozen actor_bc_flow that must never "
                "receive an update. Zero-gradient alone keeps plain Adam from moving it (same "
                "mechanism as target_critic), but AdamW's decoupled weight decay is NOT "
                "gradient-gated -- it would silently decay these loaded params toward zero every "
                "step. Use weight_decay=0 for flow_rejection runs."
            )
            actor_bc_flow_def = ActorVectorField(
                hidden_dims=config['bc_actor_hidden_dims'],
                action_dim=full_action_dim,
                layer_norm=False,  # explicit -- flow BC policy is never layer-normed
                encoder=encoders.get('actor_bc_flow'),
            )
            ex_times = ex_actions[..., :1]
            network_info['actor_bc_flow'] = (actor_bc_flow_def, (ex_observations, full_actions, ex_times))
        elif config['policy_method'] == 'awr':
            actor_def = Actor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=full_action_dim,
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

        params = network.params
        params['modules_target_critic'] = params['modules_critic']

        if config['policy_method'] == 'flow_rejection':
            bc_actor_params = load_frozen_bc_flow_params(
                config['bc_checkpoint'], ex_observations, ex_actions,
                horizon_length=config['horizon_length'],
                actor_hidden_dims=config['bc_actor_hidden_dims'],
                seed=seed,
            )
            loaded_shape = jax.tree_util.tree_map(lambda x: x.shape, bc_actor_params)
            expected_shape = jax.tree_util.tree_map(lambda x: x.shape, params['modules_actor_bc_flow'])
            assert loaded_shape == expected_shape, (
                f"BC checkpoint at {config['bc_checkpoint']!r} has architecture {loaded_shape}, "
                f"expected {expected_shape} -- check bc_actor_hidden_dims/horizon_length match "
                "what the checkpoint was actually trained with."
            )
            params['modules_actor_bc_flow'] = bc_actor_params

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='aciql',  # Agent name.
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
            alpha=10.0,  # AWR inverse temperature (higher = more greedy toward Q). Same
            # non-standard direction as iql.py's alpha, opposite FQL's BC-strength alpha.
            expectile=0.9,  # Expectile for the V regression (higher = closer to in-sample max).
            num_qs=2,  # Critic ensemble size.
            const_std=True,  # Whether the actor uses a fixed (vs. learned) standard deviation.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, ...).
            horizon_length=ml_collections.config_dict.placeholder(int),  # Chunk length AND n-step
            # return length -- set by main.py, tied together by construction (matches acfql.py).
            weight_decay=0.,  # Weight decay.
            truncate_reward_at_success=False,  # Once a step within the sampled h-step window
            # reaches success (mask==0), freeze the reward sum there instead of continuing to
            # accumulate whatever the raw (non-goal-directed) play data does afterward. Crucial for
            # chunked/n-step targets specifically -- see utils/datasets.py's sample_sequence.
            policy_method='awr',  # 'awr' | 'flow_rejection'. 'flow_rejection' loads a pretrained,
            # frozen agents/bc.py flow-BC checkpoint (config['bc_checkpoint'], required) and
            # extracts chunks via best-of-N rejection sampling against this agent's own
            # already-trained critic, instead of AWR -- see class docstring for why this policy is
            # pretrained externally rather than trained jointly here. value_loss/critic_loss are
            # unaffected either way.
            bc_checkpoint=ml_collections.config_dict.placeholder(str),  # Path to a pretrained
            # agents/bc.py flow-BC params_*.pkl (policy_method='flow_rejection' only). Must have
            # been trained on this same dataset/env with horizon_length==this agent's own.
            bc_actor_hidden_dims=(512, 512, 512, 512),  # Architecture of the checkpoint at
            # bc_checkpoint -- must match exactly what it was trained with (flow_rejection only).
            flow_steps=10,  # Euler integration steps for the flow BC policy (flow_rejection only).
            actor_num_samples=16,  # N candidates sampled for rejection sampling (flow_rejection only).
        )
    )
    return config
