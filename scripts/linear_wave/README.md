# The linear-wave experiment

Symplectic autoencoders (SAEs) and the proper symplectic decomposition (PSD) as reductions of the
linear wave equation, compared by their projection and reduction errors at wave speeds outside the
training set. The data set is [`linear_wave`](../../docs/src/linear_wave.md); everything below runs
in the `scripts/` environment.

| script | does | writes to `LW_OUTPUT_DIR` |
|:--|:--|:--|
| `train_sae.jl` | trains one SAE per reduced dimension `2n` and seed, started at the PSD of that dimension | `weights/sae_nXX_seedS_eEEEE_warm.h5` (weights, loss history, settings, package versions), after `LW_N_EPOCHS` epochs and at every epoch count in `LW_CHECKPOINTS` |
| `errors.jl` | fits the PSD of each dimension, and computes the projection and reduction errors of every reduction at the test wave speeds, with the reduced model of `HRedSys` integrated by the implicit midpoint rule | `errors$(LW_TAG).csv`, `errors$(LW_TAG).h5` |
| `plot_waves.jl` | the FOM solution for three wave speeds at three times | `wave_plot.h5`, `wave_plot.pdf` |
| `plot_errors.jl` | the errors against `n`, one panel per test wave speed, from `errors$(LW_TAG).h5` | `errors$(LW_TAG).pdf` |

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
several processes can share the training by splitting the dimensions (or the seeds) between them:

```sh
for d in 2,6,10,14 3,7,11,15 4,8,12 5,9,13; do
    LW_HALF_REDUCED_DIMS=$d julia --project=scripts scripts/linear_wave/train_sae.jl &
done; wait
```

With four such processes on a 14-core Apple M-series CPU, 1000 epochs of one network took about
13 minutes, so the 14 networks of the default settings took about an hour; `errors.jl` took about
another hour.

## The warm start

The SAE contains the PSD: with the scale of every gradient layer set to zero, the gradient layers
are the identity and the network is its two PSD-like layers, the PSD. `warm_start!` starts each
SAE there, with the Stiefel matrices of the PSD of its dimension, so training starts from the PSD's
error and only has to improve on it. Started at random instead (`LW_WARM_START=0`), the SAE of
2n = 30 stayed above the PSD's training error after 100 epochs, which is an optimization
shortfall, not a limit of the architecture. With the warm start, the seed decides only the inactive
weights of the gradient layers, so one seed is the default.

1000 epochs against 100 (both saved with `LW_CHECKPOINTS=100,200,500`): the projection errors at the
test wave speeds hardly change, but for 2n ≥ 22 the reduction errors fall by up to a factor of two.

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
| `LW_SEEDS` | `1` | seeds of the SAE initialization |
| `LW_WARM_START` | `1` | start every SAE at the PSD of its dimension |
| `LW_N_EPOCHS` | `1000` | training epochs |
| `LW_CHECKPOINTS` | `LW_N_EPOCHS` | epoch counts after which the network is saved as well |
| `LW_TAG` | empty | appended to the names of the error files |
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
