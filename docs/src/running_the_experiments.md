# Running the Experiments

There are two families of scripts, and which one you want depends on which package you are
exercising.

`scripts/gml/` is written against `GeometricMachineLearning`, so it gets `DataLoader`,
`ClassificationTransformer`, `NeuralNetwork` and the optimizers from the library and each script is a
few dozen lines. `transformer_mnist.jl` is the one the [MNIST Tutorial](@ref) walks through;
`transformer_fashion_mnist.jl` is the same thing on Fashion-MNIST. Both write their four loss
arrays, wall-clock times and test accuracies to a `.jld2`, and `plot_mnist_results.jl` draws the two
loss-curve figures from that file:

```
julia --project=scripts scripts/gml/transformer_mnist.jl
julia --project=scripts scripts/gml/plot_mnist_results.jl mnist_parameters.jld2
```

The plotting is separate from the training for the same reason the figures of [The Numerical
Experiment on Homogeneous Spaces](@ref) are drawn from checked-in CSVs: the run is four
configurations of 500 epochs, and redrawing a figure must not cost a rerun of it. The argument
defaults to `mnist_parameters.jld2`, and the output names follow it, so passing
`fashion_mnist_parameters.jld2` produces `fashion_mnist_*.png` rather than overwriting the MNIST
figures.

`scripts/geometric_optimizers/` is written against `GeometricOptimizers` *alone*.
`GeometricMachineLearning` depends on `GeometricOptimizers`, so the dependency cannot be inverted and
none of the library's neural network machinery is available: the network, the loss, the
initialization and the batching are written out explicitly, which is why these scripts are ten times
longer. They are what produced the figures on [The Numerical Experiment on Homogeneous
Spaces](@ref). This page is about them.

Run them with the scripts environment:

```
julia --project=scripts scripts/geometric_optimizers/mnist.jl
```

## The four configurations, and why one of them must fail

Three of them put the attention projections on the Stiefel manifold and differ only in the optimizer
— `Adam`, gradient descent, momentum — so they compare *optimizers*. The fourth, **regular weights
with `Adam`**, compares something else: it is the **baseline**, the same network with the projections
left unconstrained, and it is *expected* to sit at chance with a flat loss.

That is the published result [brantner2023generalizing](@cite): the vision transformer with
unconstrained projections and none of the usual heuristics — layer normalization, dropout,
regularization, pre-training — "is not able to learn much, as the error rate is stuck at around 1.34,
which indicates a trivial prediction". With ``L = 16`` blocks and nothing normalizing between them
the gradient that reaches the early blocks vanishes, so the network collapses onto a constant
prediction ``e_i`` and stays there. Constraining the projections is what removes the problem:
``Y^TY = \mathbb{I}``, so a block neither amplifies nor damps what passes through it.

The `1.34` follows from the loss. `network_loss` is ``\|\mathrm{pred} - \mathrm{out}\|/\|\mathrm{out}\|``,
and over a batch of ``k`` one-hot targets a constant prediction is wrong on 9 of 10 images and off by
``\sqrt{2}`` on each, so the loss is ``\sqrt{2 \cdot 0.9 \cdot k}/\sqrt{k} = \sqrt{1.8} \approx 1.342``.

**So a flat loss and an accuracy of ``\approx 0.10`` in that configuration are the experiment
working, not a bug.** They were briefly mistaken for an `Adam` defect on the non-manifold path;
that reading is wrong and the numbers are the expected ones to three digits. `mnist_metal_short.jl`
therefore judges this configuration against the plateau rather than against an accuracy floor.

## The host script

`mnist.jl` builds the same network as
`ClassificationTransformer(dl; n_heads = 7, L = 16, add_connection = false, Stiefel = Stiefel)` and
runs the same four trainings. It stays at `n_epochs = 5`, because on the host a step costs ≈5 s and
500 epochs of 29 batches would be ≈20 h *per configuration*. Points worth knowing:

- **The learning rate comes from the line search.** The optimizer *methods* only produce a direction,
  so the step size is `linesearch = Static(η)`, which is also what `Optimizer` defaults to for
  `GradientMethod`, `MomentumMethod` and `Adam`. `Adam`'s positional argument is the element type,
  not `η`, and `MomentumMethod(α)`'s `α` is the momentum coefficient, not a step size. The scripts
  pass `Static(learning_rate)` explicitly, so the rate is readable at the call site rather than being
  a package default.
- **`Zygote`, not `ForwardDiff`.** The cost of `GradientAutodiff` scales with the number of
  parameters, of which there are 154938 here, so a hand-written `∇F!` is passed to the `Optimizer`.
  Measured at the full configuration (``L = 16``, batch size 2048, `Float32`): a loss evaluation
  takes 0.84 s, a `Zygote` gradient 3.6 s (21 s for the first one, including compilation), a full
  optimizer step about 5 s.
- **`check_gradient` uses a directional derivative.** A central difference is useless here: in
  `Float32`, with `d` a random unit vector in 154938 dimensions, the directional derivative is of the
  order ``\|g\|/\sqrt{n} \approx 10^{-4}`` while the cancellation error of the difference quotient is
  ``\varepsilon(F)/2\epsilon \approx 5\cdot10^{-5}`` — the original check reported a meaningless
  4.1 %. The check uses `ForwardDiff.derivative` along `d`, which needs a single dual number, and
  reports a relative error of **4.9e-6** at the full configuration.

## The CUDA script

`mnist_cuda.jl` is the same script with the network on the GPU and `n_epochs = 500`. `CUDA` *is* part
of `scripts/Project.toml`: it installs, though non-functional, on machines without a CUDA device, so
the environment resolves everywhere and `mnist_cuda.jl` falls back to the host where there is none.

**The parameters stay on the host.** The optimizer interface of `GeometricOptimizers` cannot hold GPU
arrays — see its changelog for the two reasons — which is a regression against
`GeometricMachineLearning`, whose optimizers did run on `CUDABackend()`. It costs little here: the
optimizer touches only the 154938 parameters, ≈1.2 MB up and down per step, against ≈3 GB of
device-side activations. What is on the device is the batch, `predict`, the loss and the `Zygote`
gradient; the geometry — retraction, global section, `Adam` — runs on the host. Verified at a reduced
configuration (``L = 2``, batch size 64, 2 epochs, 256 images) against `mnist.jl`: the unconstrained
`Adam` run, the one without a random `GlobalSection`, is identical digit for digit, and the Stiefel
runs agree to `1e-5`, i.e. to the size of the global-section noise.

**The run reports to files, not to a terminal.** A workstation is reached over ssh, the run outlives
the session that started it, and 58000 optimizer steps are not something anybody watches. So the
script writes a report — the environment, the gradient checks, one line per epoch, a summary and a
verdict per configuration — and a CSV with one row per optimizer step, flushed per line and per epoch
respectively; the `.jld2` is rewritten after every configuration rather than once at the end. A run
that dies in the third configuration therefore leaves the first two complete and every finished epoch
of the third, and both text files are readable without Julia. A configuration that throws is caught,
reported with its backtrace, and does not take the remaining ones down; a non-finite loss ends that
configuration early.

`MNIST_N_EPOCHS`, `MNIST_BATCH_SIZE`, `MNIST_ACCURACY_EVERY`, `MNIST_REPORT`, `MNIST_LOSSES`,
`MNIST_OUTPUT` and `MNIST_PROGRESS` override the schedule and the paths; `run_mnist_cuda.sh` wraps
all of it for `screen`, stamps the output files with the start time and prints what to copy off the
machine.

The test accuracy and ``\|Y^TY-\mathbb{I}\|`` are measured every 25 epochs rather than once at the
end. Both cost well under a percent of the run, and they answer what a single final number cannot:
whether a configuration was still improving when it stopped, and whether the drift off the manifold
accumulates with the step count or settles. That series is what
[Drift off the manifold](@ref) plots.

## Repetitions: one configuration several times

**A single accuracy from an `Adam` configuration is a sample, not a number.** Constrained `Adam` does
not reproduce run to run: at the first step its update is ``\hat{m}/(\sqrt{\hat{v}}+\delta) \approx
\mathrm{sign}(g)`` per coordinate, and the gradient has vanished through the 16 unnormalized blocks,
so coordinates sitting at ``\approx 0`` have their sign decided by the last ulp. Two host runs with
the same seed, one epoch, gave accuracies `0.3761` and `0.3302` and drifts `4.6e-04` and `3.7e-05`
off the manifold, while `GradientMethod` and `MomentumMethod` reproduced to four decimals. A small
gap against a published number is therefore not a regression, and an accuracy meant to be quoted
comes from here.

`mnist_cuda_repetitions.jl` trains *one* configuration `MNIST_REPETITIONS` times and ends on a mean
and a corrected sample standard deviation over the repetitions — test accuracy, final epoch loss,
``\|Y^TY-\mathbb{I}\|`` and wall clock — with the individual samples printed next to each. Everything
below the run loop is `mnist_cuda.jl`'s code, with the seed turned into an argument of `train`, so a
repetition here is comparable to the corresponding run there. Each repetition builds its own
`Optimizer` and `OptimizerState`, so no cache and no iteration counter survives from the previous one.

- `MNIST_REPETITIONS` (default 5) is how often each configuration is trained. One repetition is one
  configuration of the full run, i.e. ≈1:35 h for `Adam` on an RTX 4090, so the default is ≈8 h —
  about what `mnist_cuda.jl` takes for all four.
- `MNIST_CONFIGURATIONS` (default `adam-stiefel`) selects which ones, from `adam-stiefel`,
  `adam-regular`, `gradient`, `momentum`, or `all`. Unknown keys fail before MNIST is loaded.
- `MNIST_VARY_SEED` (default `1`) gives repetition ``r`` the seed ``\mathrm{seed} + r - 1``, so the
  spread is that of the method over initializations *and* over the nondeterminism, which is the
  number to quote. With `0` every repetition uses the same seed and the spread is the nondeterminism
  alone.
- The remaining variables, the three output files and the `screen` wrapper
  (`run_mnist_repetitions.sh`, with `--smoke`, `--repeat N` and `-c LIST`) are those of
  `mnist_cuda.jl` and `run_mnist_cuda.sh`.

Every repetition is judged individually by the same `verdict`, and the statistics are reported
alongside those verdicts rather than instead of them: five repetitions whose mean clears the accuracy
floor but of which one collapsed is not the same outcome as five that all worked. A repetition that
throws is missing from the statistics, and the count of samples printed with them says so.

Its loss CSV has one column more than `mnist_cuda.jl`'s (`repetition`, after `configuration`), so
`distill_mnist_results.jl` — which feeds the figures of this documentation — does not read it. Those
figures compare the four configurations and stay the job of `mnist_cuda.jl`; this script answers how
far the `Adam` number in them can be trusted.

## The Metal script

`mnist_metal.jl` is the same script again for the GPU of an Apple silicon Mac. The host/device split,
the network and the initialization are those of `mnist_cuda.jl`.

**`Metal` is deliberately not a dependency of `scripts/Project.toml`.** It cannot be *resolved* on
Linux, and a project that lists it leaves the environment unprecompilable there — which is how the
first attempt at `mnist_cuda.jl` on an RTX 4090 died, with a `KeyError: ... "Metal"` out of
`Base.Precompilation.scan_deps!`. A Mac adds it once:

```
julia --project=scripts -e 'using Pkg; Pkg.add("Metal")'
```

That writes `Metal` back into `scripts/Project.toml`, and the line does not belong in a commit.

### Device memory

This is the one substantial difference to the CUDA script, and it is not optional: **without it the
script takes the whole machine down.** An `MtlArray` needs *two* independent things to be released,
and neither works on its own:

1. **`Metal.@autoreleasepool`.** A kernel launch autoreleases its command buffer, and a command
   buffer retains every buffer it references. In a script the outermost pool never drains, so the
   command buffers of the forward and backward passes accumulate there and pin every intermediate. No
   amount of garbage collection frees them — the reference that keeps them alive is on the
   Objective-C side.
2. **`GC.gc(false)`.** Conversely the pool can only drop a buffer once the `MtlArray` owning it has
   been finalized, and Julia's collector is driven by the size of the *Julia* heap: an `MtlArray` is
   a few hundred bytes of Julia object in front of hundreds of megabytes of Metal buffer, so it does
   not fire anywhere near often enough by itself.

Measured over ten gradient steps at a batch size of 512 (M4 Max, `Metal` 1.10), reading
`Metal.device().currentAllocatedSize` — reproduce with `metal_memory_probe.jl`:

| per step | device memory |
| --- | --- |
| neither | `0.5` → `4.2` GB, climbing by `0.41` GB per step |
| `GC.gc(true)` | `0.4` → `4.1` GB, i.e. a *full* collection changes nothing |
| pool only | up to `12.4` GB between the collections that happen to occur |
| **pool and `GC.gc(false)`** | **`0.31` GB, flat, at no measurable cost in time** |

Every batch, every chunk of `accuracy` and the gradient check therefore run inside a `device_scope`,
which is exactly those two things plus a budget check.

**Why this is worse than it sounds.** The memory of an Apple GPU is unified and an allocation does
not *fail* when it runs out: the kernel starts compressing, then paging, and the process is jetsammed
long before Metal reports anything. So the usual catch-an-`OutOfMemoryError`-and-retry-with-a-GC
safety net — `Metal.alloc_buffer_with_retry`, and the whole design of the CUDA memory pool — never
triggers. At a batch size of 2048 the leak added ≈1.6 GB per step and froze a 36 GB machine three
times before it was found; the evidence is a `JetsamEvent` in `/Library/Logs/DiagnosticReports`
naming `julia` as `largestProcess`, *not* a kernel panic. `Metal` 1.10 does have a `maybe_collect` in
`src/memory_pressure.jl`, but it only engages above 75 % of `recommendedMaxWorkingSetSize` — 27 GB
here — and only calls `GC.gc(false)`, which is useless against an Objective-C retain.

Because the failure mode is silent, the script also carries a `memory_budget`, two thirds of the
recommended working set, checked at the *in-step* high water mark inside `∇F!`. A leak that comes
back stops the script with an error instead of the machine.

Verified over a full epoch at the full configuration (batch size 2048, 29 batches): device memory
stays at `0.01`–`0.05` GB between steps, the in-step peak is `9.4` GB against the `18` GB budget,
`check_gradient` reports `6.1e-8` and the loss falls from `1.06` to `0.86`. A step takes ≈2 s, so the
four full runs would be ≈32 h on an M4 Max — against ≈15 h on an RTX 4090 at ≈1.0 s/step.

### The short Metal run

`mnist_metal_short.jl` exists because the full run is a day and a half, and a day and a half should
not be started on the strength of three epochs. It runs the same four trainings on a schedule
*derived* at runtime: after the gradient check has compiled the forward and backward passes it times
a handful of real `Adam` steps, divides the wall clock it has been given (`MNIST_TIME_BUDGET`, two
hours by default) by the four configurations and picks the `n_epochs` that fits; a per-run deadline
truncates a configuration that overshoots its share so the later ones still get their time. It
therefore finishes inside its budget on any machine.

It checks what a short run can check: the gradient against a host directional derivative for both
initializations, that the three constrained configurations reduce their loss and classify above a
floor, that the unconstrained one sits at the plateau of `1.342` it is supposed to sit at, that the
parameters stay finite and on the manifold, and that device memory stays inside its budget. It ends
on a verdict per configuration rather than a table to be read by eye. What it cannot answer is the
accuracy question: a few hundred steps are a few per cent of the schedule.

## References

```@bibliography
Pages = ["running_the_experiments.md"]
Canonical = false

brantner2023generalizing
```
