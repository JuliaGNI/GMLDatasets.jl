# Settings and helpers shared by the linear-wave scripts. Every setting is read from `ENV` with the
# recorded default, so that a run is described completely by its environment, and `settings()` is
# what every output file stores next to its numbers.
#
# `LW_SMOKE=1` replaces every default by a token size, so that CI runs the scripts end to end in
# seconds. A smoke run shows only that the scripts execute; its numbers mean nothing.

using GeometricMachineLearning
using GeometricMachineLearning: solve!
using GMLDatasets
using Printf
using Random

import AbstractNeuralNetworks: load, save
import GeometricProblems.LinearWave as LinearWave
import HDF5
import Pkg

const SMOKE = parse(Bool, get(ENV, "LW_SMOKE", "0"))

_setting(name, full, smoke) = get(ENV, name, SMOKE ? smoke : full)
function _ints(s)
    occursin(':', s) ? collect(range(parse.(Int, split(s, ':'))...)) :
    parse.(Int, split(s, ','))
end
_floats(s) = parse.(Float64, split(s, ','))

# The problem: N interior points, twenty wave speeds in [5/12, 2/3], 200 snapshots on [0, 1].
const N = parse(Int, _setting("LW_N", "128", "16"))
const MU_MIN = 5 / 12
const MU_MAX = 2 / 3
const N_PARAMS = parse(Int, _setting("LW_N_PARAMS", "20", "3"))
const N_TIME_STEPS = parse(Int, _setting("LW_N_TIME_STEPS", "200", "11"))
const TIMESPAN = (0.0, 1.0)
const TRAINING_MU = collect(range(MU_MIN, MU_MAX; length = N_PARAMS))
# The four test values, none of them in the training set.
const TEST_MU = _floats(_setting("LW_TEST_MU", "0.47,0.51,0.55,0.625", "0.47,0.625"))

# The reductions: one network per reduced dimension 2n and seed. With the warm start below, the
# seed only decides the random weights of the gradient layers, which start out inactive, so one
# seed is the default.
const HALF_REDUCED_DIMS = _ints(_setting("LW_HALF_REDUCED_DIMS", "2:15", "2:3"))
const SEEDS = _ints(_setting("LW_SEEDS", "1", "1"))

# `LW_WARM_START=1` starts every SAE at the PSD of its dimension (see `warm_start!`).
const WARM_START = parse(Bool, get(ENV, "LW_WARM_START", "1"))

# Training: Adam with η = 10⁻³, β₁ = 0.9, β₂ = 0.99, δ = 10⁻⁸ and the Cayley retraction, in Float64.
const N_EPOCHS = parse(Int, _setting("LW_N_EPOCHS", "1000", "2"))
# Epochs after which `train_sae.jl` also saves the network, e.g. "100,200,500"; N_EPOCHS always is.
const CHECKPOINTS = sort(unique([_ints(_setting("LW_CHECKPOINTS", string(N_EPOCHS), "1"));
                                 N_EPOCHS]))
# Appended to the names of the files `errors.jl` writes and `plot_errors.jl` reads.
const TAG = get(ENV, "LW_TAG", "")
const BATCH_SIZE = parse(Int, _setting("LW_BATCH_SIZE", "512", "16"))
const STEP_SIZE = parse(Float64, _setting("LW_STEP_SIZE", "1e-3", "1e-3"))
const ADAM_BETA1 = 0.9
const ADAM_BETA2 = 0.99
const ADAM_DELTA = 1e-8

const OUTPUT_DIR = get(ENV, "LW_OUTPUT_DIR",
    joinpath(@__DIR__, SMOKE ? "output_smoke" : "output"))

function settings()
    (N = N, mu_min = MU_MIN, mu_max = MU_MAX, n_params = N_PARAMS,
        n_time_steps = N_TIME_STEPS, timespan = collect(TIMESPAN),
        test_mu = TEST_MU, half_reduced_dims = HALF_REDUCED_DIMS, seeds = SEEDS,
        n_epochs = N_EPOCHS, warm_start = WARM_START, batch_size = BATCH_SIZE,
        step_size = STEP_SIZE,
        adam_beta1 = ADAM_BETA1, adam_beta2 = ADAM_BETA2, adam_delta = ADAM_DELTA,
        precision = "Float64", integrator = "ImplicitMidpoint", smoke = SMOKE)
end

"""
    architecture(n)

The symplectic autoencoder from the full dimension `2N` to the reduced dimension `2n`. The encoder is
four gradient layers of width `2N` (upscaled five times) and one PSD-like layer; the decoder is two
gradient layers of width `2n`, one PSD-like layer and one gradient layer of width `2N`.
"""
function architecture(n::Integer)
    SymplecticAutoencoder(2N, 2n;
        n_encoder_layers = 4, n_encoder_blocks = 2,
        n_decoder_layers = 2, n_decoder_blocks = 2,
        n_decoder_output_layers = 1,
        sympnet_upscale = 5, activation = tanh)
end

function sae_path(n, seed, epochs = N_EPOCHS)
    joinpath(OUTPUT_DIR, "weights",
        @sprintf("sae_n%02d_seed%d_e%04d%s.h5", n, seed, epochs, WARM_START ? "_warm" : ""))
end

errors_path(ext) = joinpath(OUTPUT_DIR, "errors" * TAG * "." * ext)

"The training data: the snapshots of all `TRAINING_MU`, as a `2N × n_t × n_μ` array."
function training_data()
    linear_wave_snapshots(linear_wave(;
        N = N, μ = TRAINING_MU, timespan = TIMESPAN, n_time_steps = N_TIME_STEPS))
end

"""
    fit_psd(data, n)

The proper symplectic decomposition `PSDArch(2N, 2n)` of the training data. It is fitted by an SVD
with `solve!`, so there is nothing to train and nothing random: it is recomputed wherever it is
needed rather than stored.
"""
function fit_psd(data, n::Integer)
    psd = NeuralNetwork(PSDArch(2N, 2n), CPU(), Float64)
    solve!(psd, data)
    psd
end

"""
    warm_start!(network, psd)

Start the SAE `network` at the PSD `psd` of the same dimensions: its two PSD-like layers take the
Stiefel matrices of the PSD, and every gradient layer gets the scale `0`, which makes it the
identity. The network is then the PSD, so training starts from the PSD's error, and the gradient
with respect to the scales is not zero, so the gradient layers become active as it proceeds.
"""
function warm_start!(network, psd)
    psd_weights = [p.weight for p in values(psd.params)]
    k = 0
    for (layer, p) in zip(network.model.layers, values(network.params))
        if layer isa PSDLayer
            k += 1
            p.weight.A .= psd_weights[k].A
        elseif hasproperty(p, :scale)
            p.scale .= 0
        end
    end
    k == length(psd_weights) ||
        error("expected $(length(psd_weights)) PSD-like layers, found $k")
    network
end

"The full-order problem for one test value `μ`, on the time grid of the training data."
function full_problem(μ)
    ic = LinearWave.initial_condition(μ, N)
    LinearWave.hodeproblem(N, ic.q, ic.p; parameters = (μ = μ,), timespan = TIMESPAN,
        timestep = (TIMESPAN[2] - TIMESPAN[1]) / (N_TIME_STEPS - 1))
end

"""
    provenance()

The versions every number depends on: Julia, the git commit of this repository, and the version
and source of every package in the active environment (a `[sources]` entry shows as its URL and
revision).
"""
function provenance()
    commit = try
        readchomp(`git -C $(@__DIR__) rev-parse HEAD`)
    catch
        "unknown"
    end
    dirty = try
        !isempty(readchomp(`git -C $(@__DIR__) status --porcelain`))
    catch
        true
    end
    packages = Dict{String, String}()
    for (_, info) in Pkg.dependencies()
        info.is_direct_dep || continue
        source = info.git_source === nothing ? "" :
                 " ($(info.git_source)#$(info.git_revision))"
        packages[info.name] = string(info.version) * source
    end
    (julia = string(VERSION), gmldatasets_commit = commit, gmldatasets_dirty = dirty,
        packages = packages)
end

"Write `settings()` and `provenance()` as attributes of an open HDF5 file or group."
function write_metadata(file)
    attributes = HDF5.attributes(file)
    for (k, v) in pairs(settings())
        attributes[string(k)] = v isa Bool ? Int(v) : v
    end
    p = provenance()
    attributes["julia"] = p.julia
    attributes["gmldatasets_commit"] = p.gmldatasets_commit
    attributes["gmldatasets_dirty"] = Int(p.gmldatasets_dirty)
    for (name, version) in p.packages
        attributes["package_" * name] = version
    end
    nothing
end
