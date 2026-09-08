"""Thin wrapper around generate_pointmaze_noisy.py: generates a 'navigate' dataset on the
walls-free open room (pointmaze-empty-v0, registered in envs/custom_empty_maze.py) using the
exact same generation logic (init/goal sampling, oracle-direction actor, goal-resampling-on-
success, noise machinery) as every other pointmaze navigate dataset -- only the env differs, and
this file adds none of its own dataset-generation logic. Overrides only the two flag defaults
that matter for this env (--env_name, --dataset_type); every other flag from
generate_pointmaze_noisy.py (--noise_type, --behavior_noise, --num_episodes, ...) still applies
and can still be overridden on the command line as usual.
"""
import envs.custom_empty_maze  # noqa: registers pointmaze-empty-v0
from data_gen_scripts.generate_pointmaze_noisy import FLAGS, app, main

if __name__ == '__main__':
    FLAGS.set_default('env_name', 'pointmaze-empty-v0')
    FLAGS.set_default('dataset_type', 'navigate')
    app.run(main)
