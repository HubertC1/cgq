import envs.custom_empty_maze  # noqa: F401 -- registers pointmaze-empty-v0 + singletask variants,
# the same way `import ogbench` (which pulls in ogbench.locomaze) registers the upstream maze envs.
# Runs on any `import envs` / `import envs.<submodule>`, so main.py's use of
# envs.ogbench_utils.make_ogbench_env_and_datasets picks these ids up with no further changes.
# import envs.custom_empty_maze3d  # noqa: F401 -- registers pointmaze3d-empty-v0 + singletask variants.
import envs.custom_umaze  # noqa: F401 -- registers pointmaze-umaze-v0 + singletask variants.
