import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, CausalChunkQTransformer, Value
from utils.flow_policy import flow_rejection_sample_actions, load_frozen_bc_flow_params


class CurriculumTransformerQAgent(flax.struct.PyTreeNode):
    """Causal-Transformer chunked-Q critic with an annealed-mixture V target and a randomly-sampled
    cross-head consistency loss.

    One CausalChunkQTransformer (utils/networks.py) produces Q_h(s, a_{1:h}) for every h in
    config['chunk_sizes'] (e.g. every h from 1 to 20) in a single forward pass. Three loss terms:

      1. Per-head TD loss: each Q_h regressed against r^{(h)} + discount^h * V(s_{t+h}), exactly
         like every other chunked critic in this repo (agents/curriculum_qchunk.py, agents/dqc.py).
      2. V's regression target is a *mixture* over every head, not one fixed critic: weights follow
         a Laplace kernel in raw h-value space centered on a "preferred chunk length" c(p) that
         anneals linearly (or via a cosine ease) from H=max(chunk_sizes) (the longest chunk) down to
         min(chunk_sizes) (h=1) over config['anneal_steps'] training steps. Both the mixture target
         and this term's Q_h(s,a) come from target_transformer (EMA copy, config['tau']), matching
         this repo's target-network convention for whatever feeds a regression target.
      3. Cross-head consistency: the SHORTER (tail) chunk regressed toward the LONGER (original)
         chunk, matching the same "long chunks are more reliable early on" premise the mixture
         schedule itself is built on. For every shift point a (up to config['num_consistency_shifts']
         of them, shared across the batch, fresh each step -- default is unbounded, i.e. use every
         shift), Q_{b-a}(s_{t+a}, a_{a+1:b}) is regressed toward
         stopgrad((Q_b(s,a_{1:b}) - r^{(a)}) / discount^a) for EVERY valid b=a+k simultaneously --
         the n-step identity Q_b = r^{(a)} + discount^a * Q_{b-a}(shifted), solved for the shorter
         side. One forward pass rooted at shift a yields every downstream position (that's the whole
         point of the causal transformer), so batching by shift rather than by (a,b) pair gets every
         pair sharing a shift for free: for H=20 there are only 19 possible shifts, and together they
         cover all C(20,2)=190 pairs (sum_{a=1}^{19}(20-a)=190) -- cheaper AND exhaustive, no sampling needed
         at this scale.

    No actor of its own -- per CLAUDE.md's fixed-policy-extraction methodology and matching
    agents/curriculum_qchunk.py exactly, eval-time action extraction reuses a frozen, externally
    pretrained agents/bc.py flow-BC checkpoint + best-of-N rejection sampling against only the h=1
    head. This deliberately does NOT implement ACSAC's own flow-BC policy or its joint
    argmax-over-(n,h) extraction rule -- only h=1 is ever scored at eval time, matching every other
    in-sample baseline in this repo's convention of holding policy extraction fixed while the value
    objective varies.
    """

    rng: Any
    step: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff ** 2)

    def _schedule(self):
        """Returns (p, c, kernel_tau, weights). c(p) is the *preferred chunk length itself* -- a
        real value in h-units, not an abstracted "rank" -- annealing from H=max(chunk_sizes) at
        p=0 down to min(chunk_sizes) at p=1. weights is a dict {h: weight} over config['chunk_sizes'],
        the Laplace kernel exp(-|h-c(p)|/kernel_tau) evaluated at each actual head length and
        normalized to sum to 1. (An earlier version of this operated in a separate "rank" space --
        position in the sorted head list -- since chunk_sizes was originally a sparse subset of
        lengths where rank and h-value genuinely differed. Now that chunk_sizes is always every
        integer from 1 to H, rank and h coincide up to a fixed reflection, so keeping a separate
        rank concept around was pure indirection; this operates on h directly.)"""
        chunk_sizes = self.config['chunk_sizes']
        H = max(chunk_sizes)
        h_min = min(chunk_sizes)
        anneal_steps = max(self.config['anneal_steps'], 1)
        p = jnp.clip(self.step.astype(jnp.float32) / anneal_steps, 0.0, 1.0)

        if self.config['schedule_type'] == 'cosine':
            c = H - (H - h_min) * (1.0 - jnp.cos(jnp.pi * p)) / 2.0
        else:
            c = H - p * (H - h_min)

        if self.config['kernel_tau_anneal']:
            kernel_tau = self.config['kernel_tau_start'] + p * (self.config['kernel_tau_end'] - self.config['kernel_tau_start'])
        else:
            kernel_tau = self.config['kernel_tau']

        h_arr = jnp.array(chunk_sizes, dtype=jnp.float32)
        logits = -jnp.abs(h_arr - c) / kernel_tau
        w = jax.nn.softmax(logits)
        weights = {h: w[i] for i, h in enumerate(chunk_sizes)}
        return p, c, kernel_tau, weights

    def value_loss(self, batch, grad_params, q_all_target):
        """V(s) expectile-regressed toward an annealed mixture of target_transformer's heads,
        weighted by a Laplace kernel centered on the schedule's preferred chunk length c(p).

        Takes q_all_target (target_transformer's output at the original (s_t, a_{1:H}) chunk)
        rather than computing its own forward pass -- total_loss computes it once and shares it
        with consistency_loss too, which needs the exact same quantity for its (target-side) Q_b term.
        """
        chunk_sizes = self.config['chunk_sizes']
        p, c, kernel_tau, weights = self._schedule()

        target_v = 0.0
        for h in chunk_sizes:
            target_v = target_v + weights[h] * q_all_target[..., h]

        v = self.network.select('value')(batch['observations'], params=grad_params)
        adv = target_v - v
        valid = batch['valid'][..., -1]
        value_loss = (self.expectile_loss(adv, adv, self.config['expectile']) * valid).mean()

        info = {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
            'target_v_mean': target_v.mean(),
            'schedule/p': p,
            'schedule/c': c,
            'schedule/kernel_tau': kernel_tau,
        }
        for h in chunk_sizes:
            info[f'schedule/weight_h{h}'] = weights[h]
        return value_loss, info

    def critic_loss(self, batch, q_all_live):
        """Every Q_h TD-regressed against r^{(h)} + discount^h * V(s_{t+h}).

        Takes q_all_live (shape (B, H+1), every head's prediction at the original (s_t, a_{1:H})
        chunk) rather than computing its own forward pass -- total_loss computes it once and shares
        it with consistency_loss too, since both need the exact same quantity. h is always a static
        Python int here (looping over config['chunk_sizes']), so plain indexing is fine -- no need
        for all_positions' traced-index gather machinery that consistency_loss requires.
        """
        chunk_sizes = self.config['chunk_sizes']

        info = {}
        losses = []
        for h in chunk_sizes:
            q_h = q_all_live[..., h]
            next_v = self.network.select('value')(batch['next_observations'][..., h - 1, :])
            target_h = batch['rewards'][..., h - 1] + (self.config['discount'] ** h) * batch['masks'][..., h - 1] * next_v
            valid_h = batch['valid'][..., h - 1]
            loss_h = (jnp.square(q_h - target_h) * valid_h).mean()
            losses.append(loss_h)
            info[f'critic/q_h{h}_loss'] = loss_h
            info[f'critic/q_h{h}_mean'] = q_h.mean()

        losses = jnp.stack(losses)
        critic_loss = losses.sum() if self.config['head_loss_reduction'] == 'sum' else losses.mean()
        info['critic/critic_loss'] = critic_loss
        return critic_loss, info

    def consistency_loss(self, batch, q_all_target, grad_params, rng):
        """The SHORTER (tail) chunk is regressed toward the LONGER (original) chunk -- not the
        other way around. For num_consistency_shifts randomly-sampled shift points a (shared across
        the batch, fresh each step), one extra *live* forward pass rooted at s_{t+a} yields
        Q_k(s_{t+a}, a_{a+1:a+k}) for EVERY k=1..H-a simultaneously (the whole point of the causal
        transformer: one pass computes every downstream position at once), each regressed toward
        stopgrad((Q_b(s,a_{1:b}) - r^{(a)}) / discount^a) where b=a+k -- i.e. the exact n-step
        identity Q_b = r^{(a)} + discount^a * Q_{b-a}(shifted), solved for the *shorter* side
        Q_{b-a} instead of the longer side Q_b. This matches the premise the whole curriculum is
        built on: longer chunks are the more-reliable, faster-to-learn estimate early in training
        (that's exactly why value_loss's mixture anneals from long chunks toward h=1, not the
        reverse), so it's the short chunk that should be taught by the long one, not vice versa.

        Because of this direction, which network is "live" vs. "target" is the reverse of what a
        first guess might be: Q_{b-a}(shifted) is evaluated by the LIVE transformer (params=
        grad_params -- gradient flows into the short-chunk prediction), while Q_b(s,a_{1:b}) comes
        from q_all_target -- target_transformer's output at the original (s_t, a_{1:H}) chunk,
        the same quantity value_loss's mixture target already needs, computed once in total_loss
        and shared rather than recomputed here.

        Only meaningful where there's a real continuation past step a: if a terminal occurred within
        the first a steps (mask_a=0), Q_b should have already collapsed to just r^{(a)} and the
        inverted target is degenerate, not a real constraint on the tail chunk -- gated out via
        mask_a, on top of the usual valid_b gate on the full b-window.

        num_consistency_shifts caps how many distinct shifts get used per step (default effectively
        unbounded -- use every shift). Only sampled via jax.random.choice, and rng only consumed, if
        the cap actually binds (H-1 > the configured cap); the common case (H small enough to use
        every shift) needs no randomness at all.

        a is a *traced* value when sampled, so it can't index the dict
        CausalChunkQTransformer.__call__ returns (dict keys must be static Python ints) -- gathers
        below use jnp.take, which accepts traced integer indices into a statically-shaped array.
        """
        chunk_sizes = self.config['chunk_sizes']
        H = max(chunk_sizes)
        n_shifts = min(self.config['num_consistency_shifts'], H - 1)
        if n_shifts >= H - 1:
            shifts = jnp.arange(1, H, dtype=jnp.int32)  # every shift 1..H-1 -- no sampling needed
        else:
            all_shifts = jnp.arange(1, H, dtype=jnp.int32)
            idx = jax.random.choice(rng, H - 1, shape=(n_shifts,), replace=False)
            shifts = all_shifts[idx]

        j = jnp.arange(H)
        k_range = jnp.arange(1, H)  # every possible remainder length b-a, 1..H-1

        losses, weights = [], []
        for i in range(shifts.shape[0]):
            a = shifts[i]

            # Shifted (H-a)-length remainder a_{a+1:H}, left-aligned into an H-length buffer
            # (positions >= H-a are causally invisible to anything read out below, zeroed for
            # cleanliness).
            src_idx = jnp.clip(a + j, 0, H - 1)
            gathered_actions = jnp.take(batch['actions'], src_idx, axis=-2)  # (B, H, action_dim)
            shifted_actions = jnp.where((j < (H - a))[..., None], gathered_actions, 0.0)
            shifted_obs = jnp.take(batch['next_observations'], a - 1, axis=-2)  # (B, obs_dim); s_{t+a}

            shifted_q_all = self.network.select('transformer')(
                shifted_obs, actions=shifted_actions, params=grad_params, method_name='all_positions'
            )  # (B, H + 1) -- LIVE: gradient flows into the shorter/tail chunk's prediction.

            # Every valid remainder length k=1..H-a at once: b=a+k, tail=Q_k(s_{t+a}, a_{a+1:a+k}).
            valid_k = (k_range <= (H - a)).astype(jnp.float32)  # (H-1,)
            b_idx = jnp.clip(a + k_range, 0, H)  # (H-1,)
            q_tail = jnp.take(shifted_q_all, k_range, axis=-1)  # (B, H-1) -- LIVE
            q_b = jnp.take(q_all_target, b_idx, axis=-1)  # (B, H-1) -- TARGET, from the shared main-chunk pass

            r_a = jnp.take(batch['rewards'], a - 1, axis=-1)  # (B,)
            mask_a = jnp.take(batch['masks'], a - 1, axis=-1)  # (B,)
            valid_b = jnp.take(batch['valid'], jnp.clip(b_idx - 1, 0, H - 1), axis=-1)  # (B, H-1)

            discount_a = jnp.power(jnp.float32(self.config['discount']), a.astype(jnp.float32))
            target = (q_b - r_a[:, None]) / discount_a  # invert: Q_{b-a} = (Q_b - r^{(a)}) / discount^a

            weight = valid_b * mask_a[:, None] * valid_k[None, :]  # (B, H-1)
            losses.append((jnp.square(q_tail - jax.lax.stop_gradient(target)) * weight).sum())
            weights.append(weight.sum())

        total_weight = jnp.sum(jnp.stack(weights)) if shifts.shape[0] > 0 else jnp.zeros(())
        consistency_loss = jnp.sum(jnp.stack(losses)) / jnp.maximum(total_weight, 1.0) if shifts.shape[0] > 0 else jnp.zeros(())
        return consistency_loss, {
            'consistency/loss': consistency_loss,
            'consistency/n_shifts': shifts.shape[0],
            'consistency/n_pairs_covered': total_weight,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        rng, consistency_rng = jax.random.split(rng)

        # Computed once each, at the original (s_t, a_{1:H}) chunk, and shared across whichever loss
        # terms need that exact quantity: q_all_live (gradient flows) feeds critic_loss's per-head TD
        # predictions; q_all_target (target_transformer, EMA/stopgrad) feeds both value_loss's mixture
        # target and consistency_loss's (target-side, since the short chunk now regresses toward the
        # long one) Q_b term.
        q_all_live = self.network.select('transformer')(
            batch['observations'], actions=batch['actions'], params=grad_params, method_name='all_positions'
        )
        q_all_target = self.network.select('target_transformer')(
            batch['observations'], actions=batch['actions'], method_name='all_positions'
        )

        value_loss, value_info = self.value_loss(batch, grad_params, q_all_target)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        critic_loss, critic_info = self.critic_loss(batch, q_all_live)
        info.update(critic_info)

        consistency_loss, consistency_info = self.consistency_loss(batch, q_all_target, grad_params, consistency_rng)
        info.update(consistency_info)

        loss = value_loss + critic_loss + self.config['consistency_coef'] * consistency_loss
        info['total_loss'] = loss
        return loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @staticmethod
    def _update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'transformer')

        return self.replace(network=new_network, rng=new_rng, step=self.step + 1), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)

    @jax.jit
    def batch_update(self, batch):
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    @jax.jit
    def sample_actions(self, observations, rng=None, temperature=1.0):
        # temperature accepted but unused, matching every other flow-policy agent in this repo.
        del temperature
        H = max(self.config['chunk_sizes'])
        actor_bc_flow_fn = lambda o, x, t: self.network.select('actor_bc_flow')(o, x, t)

        def critic_fn(o, a):
            # a: (..., num_samples, action_dim) single-step proposals from the frozen BC-flow.
            # CausalChunkQTransformer needs a full H-length chunk; positions after 0 are causally
            # invisible to Q_1 so they're zero-padded, not real actions.
            zeros_tail = jnp.zeros(a.shape[:-1] + (H - 1, a.shape[-1]))
            full_actions = jnp.concatenate([a[..., None, :], zeros_tail], axis=-2)
            q1 = self.network.select('transformer')(o, actions=full_actions)[1]
            return q1[None, ...]  # fake ensemble dim -- flow_rejection_sample_actions expects (num_qs, ...)

        return flow_rejection_sample_actions(
            actor_bc_flow_fn, critic_fn, observations, rng,
            num_samples=self.config['actor_num_samples'],
            flow_steps=self.config['flow_steps'],
            action_dim=self.config['action_dim'],
        )

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

        chunk_sizes = tuple(config['chunk_sizes'])
        assert chunk_sizes == tuple(sorted(chunk_sizes, reverse=True)), (
            "chunk_sizes must be given largest-first."
        )
        assert 1 in chunk_sizes, (
            "chunk_sizes must include 1 -- eval-time policy extraction (sample_actions) scores "
            "candidates against the h=1 head only."
        )
        horizon = chunk_sizes[0]
        assert config['horizon_length'] == horizon, (
            f"config['horizon_length'] (={config['horizon_length']}) drives main.py's "
            f"dataset.sample_sequence length and must equal max(chunk_sizes) (={horizon})."
        )

        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]
        ex_times = ex_actions[..., :1]
        ex_action_chunk = jnp.stack([ex_actions] * horizon, axis=0)  # (horizon, action_dim)

        transformer_def = CausalChunkQTransformer(
            chunk_sizes=chunk_sizes,
            hidden_dim=config['transformer_hidden_dim'],
            num_layers=config['transformer_num_layers'],
            num_heads=config['transformer_num_heads'],
            mlp_ratio=config['transformer_mlp_ratio'],
            layer_norm=config['transformer_layer_norm'],
        )
        value_def = Value(hidden_dims=config['value_hidden_dims'], layer_norm=config['layer_norm'], num_ensembles=1)

        network_info = dict(
            transformer=(transformer_def, (ex_observations, ex_action_chunk)),
            target_transformer=(copy.deepcopy(transformer_def), (ex_observations, ex_action_chunk)),
            value=(value_def, (ex_observations,)),
        )

        assert config['bc_checkpoint'] is not None, (
            "This agent trains no actor of its own -- config['bc_checkpoint'] must point at a "
            "pretrained agents/bc.py flow-BC checkpoint (horizon_length=1), shared across baselines, "
            "for eval-time rejection sampling. See agents/curriculum_qchunk.py's class docstring."
        )
        assert config['weight_decay'] == 0., (
            "the loaded actor_bc_flow must never move -- AdamW's decoupled weight decay isn't "
            "gradient-gated and would silently decay it. Use weight_decay=0."
        )
        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['bc_actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=False,  # explicit -- flow BC policy is never layer-normed
        )
        network_info['actor_bc_flow'] = (actor_bc_flow_def, (ex_observations, ex_actions, ex_times))

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_transformer'] = params['modules_transformer']

        bc_actor_params = load_frozen_bc_flow_params(
            config['bc_checkpoint'], ex_observations, ex_actions,
            horizon_length=1,
            actor_hidden_dims=config['bc_actor_hidden_dims'],
            seed=seed,
        )
        loaded_shape = jax.tree_util.tree_map(lambda x: x.shape, bc_actor_params)
        expected_shape = jax.tree_util.tree_map(lambda x: x.shape, params['modules_actor_bc_flow'])
        assert loaded_shape == expected_shape, (
            f"BC checkpoint at {config['bc_checkpoint']!r} has architecture {loaded_shape}, "
            f"expected {expected_shape} -- check bc_actor_hidden_dims matches what it was trained with."
        )
        params['modules_actor_bc_flow'] = bc_actor_params

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        config['chunk_sizes'] = chunk_sizes

        return cls(rng, step=jnp.zeros((), dtype=jnp.int32), network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='curriculum_transformer_q',  # Agent name.
            lr=3e-4,  # Learning rate.

            ob_dims=ml_collections.config_dict.placeholder(list),   # Observation dimensions (set automatically).
            action_dim=ml_collections.config_dict.placeholder(int), # Action dimension (set automatically).

            batch_size=256,  # Batch size.

            # Head set.
            chunk_sizes=tuple(range(20, 0, -1)),  # Every chunk length 1..20, largest first. Must
            # include 1 (eval-time extraction scores h=1 only) and be given largest-first.
            horizon_length=20,  # Drives main.py's dataset.sample_sequence length; must equal
            # max(chunk_sizes). main.py sets config['horizon_length'] from FLAGS.horizon_length --
            # set ec.value.horizon_length to match chunk_sizes[0] in the exp config.

            # Transformer critic architecture.
            transformer_hidden_dim=128,
            transformer_num_layers=2,
            transformer_num_heads=8,
            transformer_mlp_ratio=4,
            transformer_layer_norm=True,

            # V network (plain MLP, untouched by the transformer).
            value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,

            discount=0.99,
            tau=0.005,  # target_transformer EMA rate.

            expectile=0.9,  # IQL expectile for V's regression toward the mixture target.
            head_loss_reduction='mean',  # 'mean' | 'sum' -- how the K per-head TD losses combine.

            # Annealed mixture schedule, in raw h-value space.
            anneal_steps=ml_collections.config_dict.placeholder(int),  # REQUIRED -- no default
            # makes sense across datasets/step budgets. p(t) = clip(step/anneal_steps, 0, 1).
            schedule_type='linear',  # 'linear' | 'cosine'. c(p) sweeps preferred chunk length
            # H=max(chunk_sizes) (longest) -> min(chunk_sizes) (h=1) as p goes 0 -> 1.
            kernel_tau=1.0,  # Laplace kernel bandwidth in h-units, w_h(p) ~ exp(-|h-c(p)|/kernel_tau).
            kernel_tau_anneal=False,  # If True, kernel_tau itself linearly anneals kernel_tau_start -> kernel_tau_end.
            kernel_tau_start=1.0,
            kernel_tau_end=1.0,

            # Cross-head consistency loss.
            consistency_coef=1.0,  # Weight on the consistency loss term.
            num_consistency_shifts=999,  # Cap on distinct shift points a used per step (each shift
            # yields every valid (a,b) pair sharing it in one extra forward pass -- see
            # consistency_loss's docstring). Default 999 effectively means "use every shift"
            # (min(this, H-1)) -- only kicks in as a real subsample for H large enough that H-1
            # shifts/step is too expensive.

            # Eval-time policy extraction: shared frozen BC-flow + best-of-N against the h=1 head
            # only. No actor is trained by this agent -- see class docstring.
            bc_checkpoint=ml_collections.config_dict.placeholder(str),  # Path to a pretrained
            # agents/bc.py flow-BC params_*.pkl, trained with horizon_length=1.
            bc_actor_hidden_dims=(512, 512, 512, 512),  # Architecture of the checkpoint at bc_checkpoint.
            flow_steps=10,          # Euler integration steps for the flow BC policy.
            actor_num_samples=32,   # N candidates for eval-time rejection sampling.
            weight_decay=0.,        # Must stay 0 -- see create()'s assertion.
        )
    )
    return config
