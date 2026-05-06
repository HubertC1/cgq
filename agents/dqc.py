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

class DQCAgent(flax.struct.PyTreeNode):
    """Decoupled Q-chunking"""

    rng: Any
    network: Any
    config: Any = nonpytree_field()


    def chunk_critic_loss(self, batch, grad_params, rng):
        """Compute the chunk critic loss."""
        rng, _ = jax.random.split(rng)

        batch_actions=jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))

        next_v = self.network.select('value')(batch['next_observations'][...,-1,:])
        
        target_v = batch['rewards'][...,-1] + \
            (self.config['discount'] ** self.config['horizon_length']) * batch['masks'][...,-1] * next_v
        q = self.network.select('chunk_critic')(
            batch['observations'],
            actions=batch_actions, params=grad_params)
        critic_loss= (jnp.square(q - target_v) * batch['valid'][..., -1]).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def action_critic_loss(self, batch, grad_params, rng):
        """Compute the action critic loss."""

        chunk_next_observations=batch["next_observations"][...,-1,:]
        batch_chunk_actions=jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        if self.config["use_chunk_critic"]:
            target_v = self.network.select('chunk_critic')(batch['observations'], actions=batch_chunk_actions)
        else:
            next_v = self.network.select('value')(chunk_next_observations)
            target_v = batch["rewards"][...,-1:] + (self.config['discount'] ** self.config['horizon_length']) * batch['masks'][...,-1] * next_v

        q = self.network.select('action_critic')(batch['observations'], 
            actions=batch['actions'].reshape(batch['actions'].shape[0], -1)[..., :self.config["ac_action_dim"]], params=grad_params)
        
        weight = jnp.where(target_v >= q, self.config['kappa_d'], (1 - self.config['kappa_d']))

        if self.config["distill_method"] == "expectile":
            critic_loss = (weight * jnp.square(q - target_v) * batch['valid'][..., -1]).mean()
        elif self.config["distill_method"] == "quantile":
            critic_loss = (weight * jnp.abs(q - target_v) * batch['valid'][..., -1]).mean()
        else:
            raise NotImplementedError

        total_loss = critic_loss
        info = {'critic_loss': critic_loss, 'q_mean': q.mean(), 'q_max': q.max(), 'q_min': q.min()}
        
        ex_actions = batch['actions'].reshape(batch['actions'].shape[0], -1)[..., :self.config["ac_action_dim"]]
        
        ex_qs = self.network.select('target_action_critic')(batch['observations'], 
            actions=ex_actions)

        if self.config['q_agg'] == "mean":
            ex_q = ex_qs.mean(axis=0)
        else:
            ex_q = ex_qs.min(axis=0)

        v = self.network.select('value')(batch["observations"], params=grad_params)

        if self.config["implicit_backup_type"] == "expectile":
            weight = jnp.where(ex_q >= v, self.config['kappa_b'], (1 - self.config['kappa_b']))
            value_loss = (weight * jnp.square(v - ex_q) * batch['valid'][...,-1]).mean()

        elif self.config["implicit_backup_type"] == "quantile":
            weight = jnp.where(ex_q >= v, self.config['kappa_b'], (1 - self.config['kappa_b']))
            value_loss = (weight * jnp.abs(v - ex_q) * batch['valid'][...,-1]).mean()
            
        else:
            raise NotImplementedError
        
        total_loss += value_loss
        info.update({"value_loss": value_loss, "adv": (ex_q - v).mean(), "v_mean": v.mean(), "v_max": v.max(), "v_min": v.min()})

        return total_loss, info

    def actor_loss(self, batch, grad_params, rng):
        # batch['actions'] : (batch_size, horizon_length, action_dim) 
        batch_size, _, _ = batch['actions'].shape
        rng, x_rng, t_rng, _ = jax.random.split(rng, 4)
        
        # BC flow loss.
        x_0 = jax.random.normal(x_rng, (batch_size, self.config["ac_action_dim"]))
        x_1 = batch['actions'].reshape(batch_size,-1)[..., :self.config["ac_action_dim"]]  
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

        rng, actor_rng, action_critic_rng, chunk_critic_rng = jax.random.split(rng, 4)

        if self.config["use_chunk_critic"]:
            chunk_critic_loss, chunk_critic_info = self.chunk_critic_loss(batch, grad_params, chunk_critic_rng)
            for k, v in chunk_critic_info.items():
                info[f'chunk_critic/{k}'] = v

        action_critic_loss, action_critic_info = self.action_critic_loss(batch, grad_params, action_critic_rng)
        for k, v in action_critic_info.items():
            info[f'action_critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = (chunk_critic_loss if self.config["use_chunk_critic"] else 0) + action_critic_loss + actor_loss
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
        self.target_update(new_network, 'action_critic')

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
        rng=None, # Change to rng
        best_of_n_override=None,
    ):
        seed = rng if rng is not None else self.rng
        """Sample actions from the actor."""
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
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_action_chunks = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        ac_action_dim = config["policy_chunk_size"] * action_dim
        ex_action_low_chunks = ex_action_chunks[..., :ac_action_dim]

        # Define critic and actor networks.
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
            action_critic=(action_critic_def, (ex_observations, ex_action_low_chunks)),
            target_action_critic=(target_action_critic_def, (ex_observations, ex_action_low_chunks)),
            actor_bc=(actor_bc_flow_def, (ex_observations, ex_action_low_chunks, ex_times)),  # unconditional BC
        )
        if config["use_chunk_critic"]:
            network_info.update(dict(chunk_critic=(chunk_critic_def, (ex_observations, ex_action_chunks)))) # High value action chunk
        network_info.update(dict(value=(value_def, (ex_observations))))

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params

        params['modules_target_action_critic'] = params['modules_action_critic']
        
        config['ob_dims'] = ob_dims
        config["action_dim"] = action_dim
        config["ac_action_dim"] = ac_action_dim # Low value action chunk, currently 1

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='dqc',   # Agent name.
            lr=3e-4,            # Learning rate.
            
            ob_dims=ml_collections.config_dict.placeholder(list),   # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int), # Action dimension (will be set automatically).
            
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Policy network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,        # Whether to use layer normalization for the critic(s).
            actor_layer_norm=True,  # Whether to use layer normalization for the policy.
            
            discount=0.999, # Discount factor.
            tau=0.005,      # Target network update rate.
            num_qs=2,       # Number of Q ensembles.
            q_agg='mean',   # Aggregation function for Q values
            flow_steps=10,  # Number of flow steps for the policy.
            
            # DQC horizon parameters
            use_chunk_critic=True,      # Whether or not to use a separate chunked critic
            horizon_length=10,          # Backing up value from a couple of steps in the future. 
                                        #   Same as the critic chunk size if use_chunk_critic is True.
            policy_chunk_size=1,        # Policy chunk size.
            
            # DQC backup and distillation parameters
            distill_method="expectile",         # Implicit maximization loss for training the distilled critic
            kappa_d=0.5,                        # Implicit coefficient for distillation

            implicit_backup_type="quantile",    # Implicit maximization loss for implicit value backup
            kappa_b=0.9,                        # Implicit value backup coefficient

            best_of_n=32,                       # Best-of-N policy extraction
        )
    )
    return config