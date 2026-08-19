# Pendulum prototype audit

The production numerical API now lives in `src/pendulum.jl`. The files in
this directory remain intentionally separate from the package module because
they combine experiments, plotting, animation, and long-running training.

## Classification

- `pendulum_4d.jl`: migrated numerical conversion and trajectory ideas;
  obsolete as a package entry point because it trains and plots at top level.
- `pendulum_4d_train_sae.jl`: retained as historical SAE experiment; the
  reusable data generation is replaced by `pendulum_dataset`.
- `pendulum_4d_train_reduced.jl`, `pendulum_4d_train_reduced_and_plot.jl`,
  `pendulum_4d_sae_reduced.jl`, and `pendulum_4d_sae_reduced_cuda.jl`:
  retained as research experiments, not package functionality.
- `pendulum_4d_bigger_network.jl`: retained as an ensemble/network-size
  experiment; its ensemble generation is covered by the public dataset API.
- `pendulum_4d_plot.jl`, `pendulum_sae_latent_plot.jl`,
  `pendulum_energy_conservation_preview.jl`, and
  `pendulum_energy_conservation_anim.jl`: retained as visualization
  prototypes; `scripts/pendulum/plot_dataset.jl` is the small reproducible
  reference workflow.
- `pendulum.jl`: retained as the original two-dimensional baseline experiment.

No prototype is deleted in this pass because the files are untracked research
material and their external references have not been exhaustively audited.
