# Atomic MERlin analysis-task definitions

One YAML file per literal MERlin analysis task. Each holds only that
task's own tunable parameters -- never the structural cross-reference
keys (`warp_task`/`preprocess_task`/`optimize_task`/`previous_iteration`/
`global_align_task`/`segment_task`/`decode_task`/`filter_task`/.../
`random_seed`). Those are injected by `build_merlin_analysis_parameters()`
(`src/MERci/acquisition/merlin_config.py`) based on which atoms are
actually present in the recipe being assembled -- hardcoding one here
would be silently overwritten (or conflict) at assembly time, so don't.

`../recipes/*.yaml` hold the explicit ordered task-name lists that
`build_merlin_analysis_parameters()` assembles these atoms into one
MERlin analysis-tasks JSON. Each atom's own file documents its own
parameters (defaults, any per-parameter notes); this README only covers
the convention shared by all of them.
