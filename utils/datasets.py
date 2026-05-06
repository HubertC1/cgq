from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict
from tqdm import tqdm


def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


@partial(jax.jit, static_argnames=('padding',))
def random_crop(img, crop_from, padding):
    """Randomly crop an image.

    Args:
        img: Image to crop.
        crop_from: Coordinates to crop from.
        padding: Padding size.
    """
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='edge')
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


@partial(jax.jit, static_argnames=('padding',))
def batched_random_crop(imgs, crop_froms, padding):
    """Batched version of random_crop."""
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)


def compute_rtg_single_gamma(rewards, terminals, discount):
    rtg = np.zeros_like(rewards)
    rtg[-1] = rewards[-1]
    for i in range(len(rewards) - 2, -1, -1):
        rtg[i] = rewards[i] + discount * rtg[i + 1] * (1 - terminals[i])
    return rtg


def compute_rtg_two_gamma_packed(rewards, terminals, gamma1, gamma2, L):
    rewards = np.asarray(rewards, dtype=float)
    terminals = np.asarray(terminals, dtype=bool)
    T = len(rewards)

    G = np.zeros(T, dtype=float)

    # Pre-compute gamma1 powers
    gamma1_powers = gamma1 ** np.arange(L)

    # Find all terminal indices
    terminal_indices = np.where(terminals)[0]

    # Process each episode separately
    episode_start = 0
    for episode_end in tqdm(terminal_indices):
        # Process each timestep in the episode
        for t in range(episode_start, episode_end + 1):
            G_t = 0.0
            elapsed = 0
            idx = t

            while idx <= episode_end:
                # Calculate block size
                block_end = min(idx + L, episode_end + 1)
                d = block_end - idx

                # Compute discounted rewards for block
                g1 = gamma1_powers[:d]
                R_block = np.dot(g1, rewards[idx:block_end])
                G_t += (gamma2**elapsed) * R_block

                # Move to next block
                idx = block_end
                elapsed += d

                if idx > episode_end:
                    break

            G[t] = G_t

        episode_start = episode_end + 1

    return G


def compute_rtg(rewards, terminals, gamma1, gamma2, L):
    if gamma1 != gamma2:
        return compute_rtg_two_gamma_packed(rewards, terminals, gamma1, gamma2, L)
    else:
        return compute_rtg_single_gamma(rewards, terminals, gamma1)


class Dataset(FrozenDict):
    """Dataset class."""

    @classmethod
    def create(cls, freeze=True, **fields):
        """Create a dataset from the fields.

        Args:
            freeze: Whether to freeze the arrays.
            **fields: Keys and values of the dataset.
        """
        data = fields
        assert 'observations' in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        self.frame_stack = None  # Number of frames to stack; set outside the class.
        self.p_aug = None  # Image augmentation probability; set outside the class.
        self.return_next_actions = False  # Whether to additionally return next actions; set outside the class.
        self.normalize_rewards = False  # Whether to normalize rewards; set outside the class.
        self.additional_normalize_rewards = False  # Whether to normalize rewards accorinding to the action sequence.
        self.additional_normalize_rewards_scale = 1.0  # Scale for additional normalization of rewards.
        self.brc_normalize_rewards = False  # Whether to normalize rewards according to brc paper.
        self.v_max = None  # Maximum value of the reward; set outside the class.
        self.actor_action_sequence = (
            1  # Actor action sequence; set outside the class. Used for outputting action sequence.
        )
        self.critic_action_sequence = 1  # Critic action sequence; set outside the class. Used for options framework.
        self.nstep = 1  # Number of steps for n-step return; set outside the class.
        self.discount = 1.0  # Discount factor; set outside the class.
        self.discount2 = 1.0  # Discount factor for actor; set outside the class.

        self.terminal_locs = np.nonzero(self['terminals'] > 0)[0]
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        self._raw_rtgs = None
        self._valid_indices = None
        self._stats = None

    @property
    def raw_rtgs(self):
        if self._raw_rtgs is None:
            self._raw_rtgs = compute_rtg(self['rewards'], self['terminals'], self.discount, self.discount2, self.nstep)
        return self._raw_rtgs

    @property
    def max_rewards(self):
        scaled_rewards = self.scale_rewards(self['rewards'])
        return np.max(scaled_rewards)

    @property
    def min_rewards(self):
        scaled_rewards = self.scale_rewards(self['rewards'])
        return np.min(scaled_rewards)

    @property
    def valid_indices(self):
        """Get valid indices."""
        if self._valid_indices is None:
            valid_indices = []
            for start, end in zip(self.initial_locs, np.append(self.terminal_locs, self.size)):
                traj_len = end - start + 1
                valid_end = start + max(0, traj_len - (self.actor_action_sequence * self.nstep - 1))
                if valid_end > start:
                    valid_indices.extend(range(start, valid_end))
            self._valid_indices = np.array(valid_indices)
        return self._valid_indices

    @property
    def stats(self):
        if self._stats is None:
            rewards = self.scale_rewards(self['rewards'])
            rtgs = compute_rtg(rewards, self['terminals'], self.discount, self.discount2, self.critic_action_sequence)

            self._stats = {
                'p1': np.percentile(rtgs, 1),
                'p99': np.percentile(rtgs, 99),
                'delta': np.percentile(rtgs, 99) - np.percentile(rtgs, 1),
                'rtgs': rtgs,
            }

        return self._stats

    def scale_rewards(self, rewards):
        if self.normalize_rewards + self.additional_normalize_rewards + self.brc_normalize_rewards > 1:
            raise ValueError(
                f'Only use one normalization method at a time. Currently using {self.normalize_rewards}/{self.additional_normalize_rewards}/{self.brc_normalize_rewards}'
            )
        if self.normalize_rewards:
            max_abs_rewards = np.max(np.abs(rewards))
            final_rewards = rewards / max_abs_rewards
        elif self.additional_normalize_rewards:
            final_rewards = rewards / self.additional_normalize_rewards_scale
        elif self.brc_normalize_rewards:
            # Normalize the reward using RTG and its maximum absolute values.
            rtgs_abs_max = np.max(np.abs(self.raw_rtgs))
            final_rewards = (rewards / rtgs_abs_max) * self.v_max
        else:
            final_rewards = rewards
        return final_rewards


    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices."""
        return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size: int, idxs=None):
        """Sample a batch of transitions."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        batch = self.get_subset(idxs)
        if self.frame_stack is not None:
            # Stack frames.
            initial_state_idxs = self.initial_locs[np.searchsorted(self.initial_locs, idxs, side='right') - 1]
            obs = []  # Will be [ob[t - frame_stack + 1], ..., ob[t]].
            next_obs = []  # Will be [ob[t - frame_stack + 2], ..., ob[t], next_ob[t]].
            for i in reversed(range(self.frame_stack)):
                # Use the initial state if the index is out of bounds.
                cur_idxs = np.maximum(idxs - i, initial_state_idxs)
                obs.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self['observations']))
                if i != self.frame_stack - 1:
                    next_obs.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self['observations']))
            next_obs.append(jax.tree_util.tree_map(lambda arr: arr[idxs], self['next_observations']))

            batch['observations'] = jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *obs)
            batch['next_observations'] = jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *next_obs)
        if self.p_aug is not None:
            # Apply random-crop image augmentation.
            if np.random.rand() < self.p_aug:
                self.augment(batch, ['observations', 'next_observations'])
        return batch

    def sample_sequence(self, batch_size, sequence_length, discount):
        idxs = np.random.randint(self.size - sequence_length + 1, size=batch_size)
        
        data = {k: v[idxs] for k, v in self.items()}

        # Pre-compute all required indices
        all_idxs = idxs[:, None] + np.arange(sequence_length)[None, :]  # (batch_size, sequence_length)
        all_idxs = all_idxs.flatten()
        
        # Batch fetch data to avoid loops
        batch_observations = self['observations'][all_idxs].reshape(batch_size, sequence_length, *self['observations'].shape[1:])
        batch_next_observations = self['next_observations'][all_idxs].reshape(batch_size, sequence_length, *self['next_observations'].shape[1:])
        batch_actions = self['actions'][all_idxs].reshape(batch_size, sequence_length, *self['actions'].shape[1:])
        batch_rewards = self['rewards'][all_idxs].reshape(batch_size, sequence_length, *self['rewards'].shape[1:])
        batch_masks = self['masks'][all_idxs].reshape(batch_size, sequence_length, *self['masks'].shape[1:])
        batch_terminals = self['terminals'][all_idxs].reshape(batch_size, sequence_length, *self['terminals'].shape[1:])
       
        # For single step critic and actor
        batch_next_observations_step = self['next_observations']
         
        # Calculate next_actions
        next_action_idxs = np.minimum(all_idxs + 1, self.size - 1)
        batch_next_actions = self['actions'][next_action_idxs].reshape(batch_size, sequence_length, *self['actions'].shape[1:])
        
        # Use vectorized operations to calculate cumulative rewards and masks
        rewards = np.zeros((batch_size, sequence_length), dtype=float)
        masks = np.ones((batch_size, sequence_length), dtype=float)
        terminals = np.zeros((batch_size, sequence_length), dtype=float)
        valid = np.ones((batch_size, sequence_length), dtype=float)
        
        # Vectorized calculation
        rewards[:, 0] = batch_rewards[:, 0].squeeze()
        masks[:, 0] = batch_masks[:, 0].squeeze()
        terminals[:, 0] = batch_terminals[:, 0].squeeze()
        
        discount_powers = discount ** np.arange(sequence_length)
        for i in range(1, sequence_length):
            rewards[:, i] = rewards[:, i-1] + batch_rewards[:, i].squeeze() * discount_powers[i]
            masks[:, i] = np.minimum(masks[:, i-1], batch_masks[:, i].squeeze())
            terminals[:, i] = np.maximum(terminals[:, i-1], batch_terminals[:, i].squeeze())
            valid[:, i] = 1.0 - terminals[:, i-1]
        
        # Reorganize observations data format - maintain the exact same shape as the original function
        if len(batch_observations.shape) == 5:  # Visual data: (batch, seq, h, w, c)
            # Transpose to (batch, h, w, seq, c) format, consistent with the original function
            observations = batch_observations.transpose(0, 2, 3, 1, 4)  # (batch_size, h, w, sequence_length, c)
            next_observations = batch_next_observations.transpose(0, 2, 3, 1, 4)  # (batch_size, h, w, sequence_length, c)
        else:  # State data: maintain (batch, seq, state_dim) shape
            observations = batch_observations  # (batch_size, sequence_length, state_dim)
            next_observations = batch_next_observations  # (batch_size, sequence_length, state_dim)
        
        # Maintain the 3D shape of actions and next_actions, consistent with the original function
        actions = batch_actions  # (batch_size, sequence_length, action_dim)
        next_actions = batch_next_actions  # (batch_size, sequence_length, action_dim)
        
        return dict(
            observations=data['observations'].copy(),
            full_observations=observations,
            actions=actions,
            masks=masks,
            rewards=rewards,
            terminals=terminals,
            valid=valid,
            next_observations=next_observations,
            next_actions=next_actions,
        )

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if self.return_next_actions:
            # WARNING: This is incorrect at the end of the trajectory. Use with caution.
            result['next_actions'] = self._dict['actions'][np.minimum(idxs + 1, self.size - 1)]
        return result

    def augment(self, batch, keys):
        """Apply image augmentation to the given keys."""
        padding = 3
        batch_size = len(batch[keys[0]])
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate([crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1)
        for key in keys:
            batch[key] = jax.tree_util.tree_map(
                lambda arr: np.array(batched_random_crop(arr, crop_froms, padding)) if len(arr.shape) == 4 else arr,
                batch[key],
            )


class ReplayBuffer(Dataset):
    """Replay buffer class.

    This class extends Dataset to support adding transitions.
    """

    @classmethod
    def create(cls, transition, size):
        """Create a replay buffer from the example transition.

        Args:
            transition: Example transition (dict).
            size: Size of the replay buffer.
        """

        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = jax.tree_util.tree_map(create_buffer, transition)
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size):
        """Create a replay buffer from the initial dataset.

        Args:
            init_dataset: Initial dataset.
            size: Size of the replay buffer.
        """

        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = jax.tree_util.tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = get_size(init_dataset)
        return dataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_size = get_size(self._dict)
        self.size = 0
        self.pointer = 0

    def add_transition(self, transition):
        """Add a transition to the replay buffer."""

        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element

        jax.tree_util.tree_map(set_idx, self._dict, transition)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = max(self.pointer, self.size)

    def clear(self):
        """Clear the replay buffer."""
        self.size = self.pointer = 0

def add_history(dataset, history_length):

    size = dataset.size
    (terminal_locs,) = np.nonzero(dataset['terminals'] > 0)
    initial_locs = np.concatenate([[0], terminal_locs[:-1] + 1])
    assert terminal_locs[-1] == size - 1

    idxs = np.arange(size)
    initial_state_idxs = initial_locs[np.searchsorted(initial_locs, idxs, side='right') - 1]
    obs_rets = []
    acts_rets = []
    for i in reversed(range(1, history_length)):
        cur_idxs = np.maximum(idxs - i, initial_state_idxs)
        outside = (idxs - i < initial_state_idxs)[..., None]
        obs_rets.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs] * (~outside) + jnp.zeros_like(arr[cur_idxs]) * outside, 
            dataset['observations']))
        acts_rets.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs] * (~outside) + jnp.zeros_like(arr[cur_idxs]) * outside, 
            dataset['actions']))
    observation_history, action_history = jax.tree_util.tree_map(lambda *args: np.stack(args, axis=-2), *obs_rets),\
        jax.tree_util.tree_map(lambda *args: np.stack(args, axis=-2), *acts_rets)

    dataset = Dataset(dataset.copy(dict(
        observation_history=observation_history,
        action_history=action_history)))
    
    return dataset


