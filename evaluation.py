from collections import defaultdict

import jax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import trange
from functools import partial


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, rng=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)

def evaluate(
    agent,
    env,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    eval_gaussian=None,
    action_shape=None,
    observation_shape=None,
    action_dim=None,
    seed=None,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.
        eval_gaussian: Standard deviation of the Gaussian noise to add to the actions.
        seed: (Optional) seed for the policy-sampling JAX key. If None, drawn from the global numpy RNG
            (the previous default behavior). Pass an explicit seed for a reproducible, isolated call --
            see evaluate_multi_seed, which also needs this to make each repeat independent.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    rng_seed = seed if seed is not None else np.random.randint(0, 2**32)
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(rng_seed))
    trajs = []
    stats = defaultdict(list)

    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        observation, info = env.reset()
            
        observation_history = []
        action_history = []
        
        done = False
        step = 0
        render = []
        action_chunk_lens = defaultdict(lambda: 0)

        action_queue = []

        gripper_contact_lengths = []
        gripper_contact_length = 0
        
        while not done:
            # To check whether the agent is flipped.

            action = actor_fn(observations=observation, temperature=eval_temperature)

            if len(action_queue) == 0:
                have_new_action = True
                action = np.array(action).reshape(-1, action_dim)
                action_chunk_len = action.shape[0]
                for a in action:
                    action_queue.append(a)
            else:
                have_new_action = False
            
            action = action_queue.pop(0)
            if eval_gaussian is not None:
                action = np.random.normal(action, eval_gaussian)

            next_observation, reward, terminated, truncated, info = env.step(np.clip(action, -1, 1))
            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                render.append(frame)

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            
            observation = next_observation
            if "proprio" in info and "gripper_contact" in info["proprio"]:
                gripper_contact = info["proprio"]["gripper_contact"]
            elif "gripper_contact" in info:
                gripper_contact = info["gripper_contact"]
            else:
                gripper_contact = None

            if gripper_contact is not None:
                if info["gripper_contact"] > 0.1:
                    gripper_contact_length += 1
                else:
                    if gripper_contact_length > 0:
                        gripper_contact_lengths.append(gripper_contact_length)
                    gripper_contact_length = 0

        if gripper_contact_length > 0:
            gripper_contact_lengths.append(gripper_contact_length)
        
        num_gripper_contacts = len(gripper_contact_lengths)

        if num_gripper_contacts > 0:
            avg_gripper_contact_length = np.mean(np.array(gripper_contact_lengths))
        else:
            avg_gripper_contact_length = 0
            
        add_to(stats, {"avg_gripper_contact_length": avg_gripper_contact_length, "num_gripper_contacts": num_gripper_contacts})

        if hasattr(env, 'get_noise_stats'):
            # Present iff the env stack includes a NoisyActionWrapper (envs/noise_wrapper.py).
            info = dict(info)
            info['noise'] = env.get_noise_stats()

        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            add_to(stats, {'episode_length': step})  # MDP steps taken this episode (until done)
            trajs.append(traj)
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders


def evaluate_multi_seed(
    agent,
    env,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    action_dim=None,
    num_seeds=1,
    base_seed=0,
):
    """Runs `evaluate` independently `num_seeds` times, each under its own reproducible seed, and
    aggregates into a mean plus a spread (std/min/max) per metric -- the fields needed to plot a
    band across eval seeds at each eval checkpoint.

    Each repeat k uses seed = base_seed + k, controlling every source of eval-time randomness:
    JAX policy sampling (passed to `evaluate`), the env's actuator-noise RNG if present
    (NoisyActionWrapper.set_noise_seed), and OGBench's internal reset-time position jitter (draws
    from the global numpy RNG). The global numpy RNG is saved before and restored after this
    function, since offline training batch sampling also draws from it (utils/datasets.py) --
    without the restore, calling this mid-training would silently perturb the sequence of
    training batches sampled afterward.

    Only the first repeat renders video (video cost is not worth multiplying by num_seeds).

    Returns:
        (agg_info, trajs, renders): agg_info has one entry per metric key from `evaluate`'s stats
        (the mean across seeds, same key name as before -- backward compatible with existing
        logging) plus '{key}_std', '{key}_min', '{key}_max' siblings. trajs/renders are from the
        first seed only.
    """
    saved_state = np.random.get_state()
    per_seed_infos = []
    trajs, renders = [], []
    try:
        for k in range(num_seeds):
            seed_k = base_seed + k
            np.random.seed(seed_k)
            if hasattr(env, 'set_noise_seed'):
                env.set_noise_seed(seed_k)
            info_k, trajs_k, renders_k = evaluate(
                agent=agent,
                env=env,
                num_eval_episodes=num_eval_episodes,
                num_video_episodes=num_video_episodes if k == 0 else 0,
                video_frame_skip=video_frame_skip,
                action_dim=action_dim,
                seed=seed_k,
            )
            per_seed_infos.append(info_k)
            if k == 0:
                trajs, renders = trajs_k, renders_k
    finally:
        np.random.set_state(saved_state)

    agg_info = {}
    for key in per_seed_infos[0].keys():
        vals = np.array([info[key] for info in per_seed_infos], dtype=np.float64)
        agg_info[key] = float(vals.mean())
        agg_info[f'{key}_std'] = float(vals.std())
        agg_info[f'{key}_min'] = float(vals.min())
        agg_info[f'{key}_max'] = float(vals.max())
    return agg_info, trajs, renders


def render_value_plot(values, slip_flags, cur_step, height, ymin, ymax):
    """A single running-V(s) line-plot frame, marker at cur_step, slip-actuated steps shaded red.
    Rendered at matplotlib's own resolution and returned as an (h, w, 3) uint8 array; the caller is
    responsible for resizing to match the env-frame panel it gets composited next to."""
    fig, ax = plt.subplots(figsize=(4, height / 100), dpi=100)
    xs = np.arange(len(values))
    for i, is_slip in enumerate(slip_flags):
        if is_slip:
            ax.axvspan(i - 0.5, i + 0.5, color='red', alpha=0.15, linewidth=0)
    ax.plot(xs, values, color='tab:blue', linewidth=1.5)
    ax.scatter([cur_step], [values[cur_step]], color='tab:orange', s=40, zorder=5)
    ax.set_xlim(0, max(len(values), 2))
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('step')
    ax.set_ylabel('V(s)')
    ax.set_title(f'step {cur_step}', fontsize=10)
    fig.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf


def render_value_replay(agent, env, seed=0, max_steps=400, temperature=0.0):
    """Roll out one episode with the agent's own eval-time policy, recording V(s_t) (via
    agent.get_value) at every step, and return a side-by-side [env frame | running V(s) curve]
    composite video -- the cube-env replacement for render_value_map.py's pointmaze-only 2D xy-grid
    value heatmap (there's no spatial grid to sweep over a high-dim manipulation observation here).
    If `env` is wrapped with envs.slip_wrapper.SlipGripperWrapper (has get_slip_stats), slip-actuated
    steps are shaded on the value curve so drops are visible next to whatever V(s) does in response.

    Used both by scripts/render_cube_value_replay.py (offline, from a checkpoint) and by main.py's
    training-loop eval (wired live, every eval_interval) -- single source of truth for both.

    Returns:
        (composite_frames, values, slip_flags): composite_frames is a uint8 array of shape
        (T, H, 2W, C), ready for get_wandb_video([composite_frames]) or imageio.mimsave.
    """
    true_action_dim = env.action_space.shape[0]
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(seed))

    observation, info = env.reset(seed=seed)
    frames, values, slip_flags = [], [], []
    action_queue = []
    prev_slip_steps = 0.0

    for _ in range(max_steps):
        v = float(np.asarray(agent.get_value(observation)))
        values.append(v)
        frames.append(np.asarray(env.render()))

        if len(action_queue) == 0:
            action = actor_fn(observations=observation, temperature=temperature)
            action = np.array(action).reshape(-1, true_action_dim)
            for a in action:
                action_queue.append(a)
        action = action_queue.pop(0)

        next_observation, reward, terminated, truncated, info = env.step(np.clip(action, -1, 1))

        if hasattr(env, 'get_slip_stats'):
            stats = env.get_slip_stats()
            slip_flags.append(stats['slip_steps'] > prev_slip_steps)
            prev_slip_steps = stats['slip_steps']
        else:
            slip_flags.append(False)

        observation = next_observation
        if terminated or truncated:
            break

    ymin, ymax = min(values), max(values)
    pad = 0.05 * (ymax - ymin + 1e-6)
    ymin, ymax = ymin - pad, ymax + pad

    composite_frames = []
    for t in range(len(frames)):
        env_frame = Image.fromarray(frames[t])
        plot_arr = render_value_plot(values, slip_flags, t, env_frame.height, ymin, ymax)
        plot_img = Image.fromarray(plot_arr).resize((plot_arr.shape[1], env_frame.height))
        composite = Image.new('RGB', (env_frame.width + plot_img.width, env_frame.height))
        composite.paste(env_frame, (0, 0))
        composite.paste(plot_img, (env_frame.width, 0))
        composite_frames.append(np.array(composite))

    return np.array(composite_frames, dtype=np.uint8), values, slip_flags


def evaluate_ant_flip(
    agent,
    env,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    eval_gaussian=None,
    action_shape=None,
    observation_shape=None,
    action_dim=None,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.
        eval_gaussian: Standard deviation of the Gaussian noise to add to the actions.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    trajs = []
    stats = defaultdict(list)

    renders = []
    last = num_eval_episodes + num_video_episodes - 1
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        observation, info = env.reset()
            
        observation_history = []
        action_history = []
        
        done = False
        step = 0
        render = []
        action_chunk_lens = defaultdict(lambda: 0)

        action_queue = []

        gripper_contact_lengths = []
        gripper_contact_length = 0
        
        ep_flipped_steps=0
        ep_up_z_history=[]

        while not done:
            # To check whether the agent is flipped. 
            z_orient = env.unwrapped.data.body("torso").xmat[8]
            ep_up_z_history.append(z_orient)
            if z_orient < 0.2:
                ep_flipped_steps += 1
            action = actor_fn(observations=observation, temperature=eval_temperature)

            if len(action_queue) == 0:
                have_new_action = True
                action = np.array(action).reshape(-1, action_dim)
                action_chunk_len = action.shape[0]
                for a in action:
                    action_queue.append(a)
            else:
                have_new_action = False
            
            action = action_queue.pop(0)
            if eval_gaussian is not None:
                action = np.random.normal(action, eval_gaussian)

            next_observation, reward, terminated, truncated, info = env.step(np.clip(action, -1, 1))
            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                render.append(frame)

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            
            observation = next_observation
            if "proprio" in info and "gripper_contact" in info["proprio"]:
                gripper_contact = info["proprio"]["gripper_contact"]
            elif "gripper_contact" in info:
                gripper_contact = info["gripper_contact"]
            else:
                gripper_contact = None

            if gripper_contact is not None:
                if info["gripper_contact"] > 0.1:
                    gripper_contact_length += 1
                else:
                    if gripper_contact_length > 0:
                        gripper_contact_lengths.append(gripper_contact_length)
                    gripper_contact_length = 0


        flipped_rate = ep_flipped_steps / step if step > 0 else 0
        is_final_flipped = 1.0 if ep_up_z_history[-1] < 0.2 else 0.0
            
        add_to(stats, {
            "flipped_rate": flipped_rate,
            "z_orient_mean": np.mean(ep_up_z_history),
            "is_final_flipped": is_final_flipped
        })
        
        if gripper_contact_length > 0:
            gripper_contact_lengths.append(gripper_contact_length)
        
        num_gripper_contacts = len(gripper_contact_lengths)

        if num_gripper_contacts > 0:
            avg_gripper_contact_length = np.mean(np.array(gripper_contact_lengths))
        else:
            avg_gripper_contact_length = 0
            
        add_to(stats, {"avg_gripper_contact_length": avg_gripper_contact_length, "num_gripper_contacts": num_gripper_contacts})

        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, render