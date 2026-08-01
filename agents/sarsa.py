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


class SARSAAgent(flax.struct.PyTreeNode):
    """On-policy SARSA agent -- estimates the *behavior* policy's value Q^{pi_beta} / V^{pi_beta}.

    Diagnostic counterpart to iql.py: where IQL's V is expectile-regressed toward Q(s, a_data) with
    expectile > 0.5 (an in-sample approximation of max_a Q(s,a)), this agent's critic bootstraps
    directly off the *actual next state-action pair logged in the dataset*
    (batch['next_actions'], computed by utils/datasets.py's sample_sequence but unused by iql.py /
    aciql.py) -- the textbook SARSA target. No max, no expectile skew, no policy-sampled action
    anywhere in the bootstrap, so Q converges to Q^{pi_beta}: the value the *data-generating* policy
    actually achieves, not the best in-sample policy's value. Comparing this agent's V against
    IQL's V (V_IQL(s) - V_SARSA(s)) is CLAUDE.md's Phase-1 suboptimality check: how much headroom
    does in-sample improvement have over the logged behavior policy at each state.

    The `value` network here is NOT part of the critic's bootstrap (unlike IQL, where V feeds the
    critic's target) -- it's a secondary, purely diagnostic head, regressed with a *fixed* expectile
    of 0.5 (plain MSE, not a config knob) toward target_critic(s, a_data), so this agent exposes a
    get_value(obs) in the same shape as iql.py's, for direct comparison (e.g. via
    evaluation.render_value_replay). Kostrikov et al. (IQL) note tau=0.5 recovers exactly this
    mean/SARSA-style backup, in contrast to tau>0.5's in-sample-max approximation.

    The actor is unweighted behavior cloning (no AWR / advantage weighting, no `alpha`) -- at eval
    time this agent's rollout should literally reproduce the behavior policy, matching what its
    critic evaluates.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def value_loss(self, batch, grad_params):
        """V(s) plain-MSE-regressed (expectile fixed at 0.5) against target_critic(s, a_data).

        Diagnostic only -- this network is never read by critic_loss's bootstrap below. It exists
        so this agent exposes get_value(obs) in the same shape as iql.py's, for a direct
        V_IQL(s) - V_SARSA(s) comparison.
        """
        actions = batch['actions'][..., 0, :]
        qs = self.network.select('target_critic')(batch['observations'], actions=actions)
        q = jnp.min(qs, axis=0) if self.config['q_agg'] == 'min' else jnp.mean(qs, axis=0)
        v = self.network.select('value')(batch['observations'], params=grad_params)
        value_loss = (jnp.square(v - q) * batch['valid'][..., -1]).mean()
        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def critic_loss(self, batch, grad_params):
        """Q(s, a) TD-regressed against the literal n-step SARSA target
        r + discount^h * mask * Q_target(s_h, a_h'), where a_h' = batch['next_actions'][..., -1, :]
        is the *actual* action the behavior policy took from s_h in the dataset -- never a sampled
        or policy-improved action. This is what makes Q converge to Q^{pi_beta} rather than Q*.
        """
        next_actions = batch['next_actions'][..., -1, :]
        next_qs = self.network.select('target_critic')(
            batch['next_observations'][..., -1, :], actions=next_actions
        )
        next_q = jnp.min(next_qs, axis=0) if self.config['q_agg'] == 'min' else jnp.mean(next_qs, axis=0)
        target_q = batch['rewards'][..., -1] + (
            self.config['discount'] ** self.config['horizon_length']
        ) * batch['masks'][..., -1] * next_q
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
        """Plain behavior cloning: BC log-prob on the dataset action, unweighted -- no AWR/advantage
        term, no `alpha`. This agent's eval-time rollout should reproduce the behavior policy
        itself, not an improved policy, so nothing here should push it off-distribution.
        """
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
        """V(s) -- diagnostic-only head, see value_loss. Same interface as iql.py's get_value, for
        a direct V_IQL(s) - V_SARSA(s) suboptimality comparison.
        """
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
            agent_name='sarsa',  # Agent name.
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
            q_agg='mean',  # Aggregation for the critic ensemble. Keep 'mean': this agent estimates
            # the behavior policy's *actual* value as an unbiased diagnostic -- 'min' (DQC/CGQ's
            # usual pessimistic control choice) would bias V_SARSA downward and inflate the apparent
            # IQL-vs-SARSA suboptimality gap.
            num_qs=2,  # Critic ensemble size.
            const_std=True,  # Whether the actor uses a fixed (vs. learned) standard deviation.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, ...).
            horizon_length=ml_collections.config_dict.placeholder(int),  # n-step SARSA return length
            # (set by main.py); use 1 for true single-step SARSA. No `alpha`/`expectile` fields --
            # both are fixed by the definition of on-policy behavior-policy evaluation, not free
            # knobs (see value_loss/actor_loss docstrings). If ec.value/ec.policy overlays a stray
            # `expectile`/`alpha`, it's silently skipped since those keys don't exist here.
            weight_decay=0.,  # Weight decay.
        )
    )
    return config
