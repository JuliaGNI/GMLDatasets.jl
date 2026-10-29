# The linear-wave experiment

Symplectic autoencoders (SAEs) and the proper symplectic decomposition (PSD) as reductions of the
linear wave equation, compared by their projection and reduction errors at wave speeds outside the
training set. The data set is [`linear_wave`](../../docs/src/linear_wave.md); everything below runs
in the `scripts/` environment.

| script | does | writes to `LW_OUTPUT_DIR` |
|:--|:--|:--|
| `train_sae.jl` | trains one SAE per reduced dimension `2n` and seed | `weights/sae_nXX_seedS.h5` (weights, loss history, settings, package versions) |
| `errors.jl` | fits the PSD of each dimension, and computes the projection and reduction errors of every reduction at the test wave speeds, with the reduced model of `HRedSys` integrated by the implicit midpoint rule | `errors.csv`, `errors.h5` |
| `plot_waves.jl` | the FOM solution for three wave speeds at three times | `wave_plot.h5`, `wave_plot.pdf` |
| `plot_errors.jl` | the errors against `n`, one panel per test wave speed, from `errors.h5` | `errors.pdf` |

`common.jl` holds the settings, the architecture and the helpers the four share.

## Running

From the repository root:

```sh
julia --project=scripts -e 'using Pkg; Pkg.update()'
julia --project=scripts scripts/linear_wave/train_sae.jl
julia --project=scripts --threads=auto scripts/linear_wave/errors.jl
julia --project=scripts scripts/linear_wave/plot_waves.jl
julia --project=scripts scripts/linear_wave/plot_errors.jl
```

`train_sae.jl` skips every network whose file exists, so it resumes after an interruption, and
several processes can share the training by splitting the seeds or the dimensions between them:

```sh
for s in 1 2 3; do LW_SEEDS=$s julia --project=scripts scripts/linear_wave/train_sae.jl & done; wait
```

With three such processes on a 14-core Apple M-series CPU a network took 70 to 100 s, so the 42
networks of the default settings took about 25 minutes.

## Settings

Every setting is an environment variable with a recorded default, and every output file stores the
settings and the package versions it was produced with.

| variable | default | meaning |
|:--|:--|:--|
| `LW_N` | `128` | interior grid points `N`; the state has `2N` components |
| `LW_N_PARAMS` | `20` | training wave speeds, equally spaced in `[5/12, 2/3]` |
| `LW_N_TIME_STEPS` | `200` | snapshots on `t ∈ [0, 1]` |
| `LW_TEST_MU` | `0.47,0.51,0.55,0.625` | test wave speeds |
| `LW_HALF_REDUCED_DIMS` | `2:15` | `n`, half the reduced dimension (`a:b` or a comma-separated list) |
| `LW_SEEDS` | `1,2,3` | seeds of the SAE initialization |
| `LW_N_EPOCHS` | `100` | training epochs |
| `LW_BATCH_SIZE` | `512` | minibatch size |
| `LW_STEP_SIZE` | `1e-3` | Adam step size; `β₁ = 0.9`, `β₂ = 0.99`, `δ = 10⁻⁸`, Cayley retraction, `Float64` |
| `LW_OUTPUT_DIR` | `scripts/linear_wave/output` | where everything is written (ignored by git) |
| `LW_SMOKE` | `0` | `1` replaces every default by a token size, as CI does; its numbers mean nothing |

The SAE is `SymplecticAutoencoder(2N, 2n; n_encoder_layers = 4, n_encoder_blocks = 2,
n_decoder_layers = 2, n_decoder_blocks = 2, n_decoder_output_layers = 1, sympnet_upscale = 5)`:
four gradient layers of width `2N` and a PSD-like layer in the encoder; two gradient layers of
width `2n`, a PSD-like layer and one gradient layer of width `2N` in the decoder.

## The errors

For a test wave speed `μ` with FOM snapshots `x⁽ᵗ⁾`, encoder `𝒫` and decoder `ℛ`, `errors.jl`
reports `GeometricMachineLearning.projection_error` and `reduction_error`,

    e_proj(μ) = √( Σₜ ‖x⁽ᵗ⁾ - ℛ(𝒫(x⁽ᵗ⁾))‖² / Σₜ ‖x⁽ᵗ⁾‖² ),
    e_red(μ)  = √( Σₜ ‖x⁽ᵗ⁾ - ℛ(x_r⁽ᵗ⁾)‖² / Σₜ ‖x⁽ᵗ⁾‖² ),

where `x_r` solves the Hamiltonian system of `H∘ℛ` from `𝒫(x⁽⁰⁾)`. With each reduction error it
reports `hred_drift`, the largest relative change of `H∘ℛ` along `x_r`, which a symplectic
integrator keeps bounded; the column `fom_drift` is the same for `H` along the FOM solution.
