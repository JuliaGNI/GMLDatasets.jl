# The `scalar-moment-adam` baseline, as one method object both trainers build.
#
# `ScalarMomentAdam` — the Adam of [li2020efficient], Algorithm 2, with its scalar second moment —
# accepts exactly one `StiefelManifold` on purpose: a *scalar* second moment is a statement about one
# manifold and means nothing pooled across a tree. Both networks in this comparison are mixed trees,
# the transformer with `StiefelManifold` attention projections beside ordinary `Matrix` and `Vector`
# leaves and the symplectic autoencoder with Stiefel PSD layers beside Euclidean SympNet ones, so the
# baseline has to be assembled *around* the released method rather than passed to it.
#
# `GeometricOptimizers.CompositeMethod` is that assembly, and it is upstream's: one method per leaf,
# `ScalarMomentAdam` on the manifold ones and ordinary `Adam` on the rest. This file is the whole of
# what is local about the baseline — the coefficients the comparison runs it at, and which ‖·‖² its
# second moment accumulates.
#
# **It exists so that there is one of it.** The two trainers used to carry an adapter each, 299 lines
# between them, reaching through two different seams: one hand-wrote a per-leaf step loop around
# `Optimizer`, the other added methods to four underscore-prefixed `GeometricMachineLearning`
# internals. They encoded the same physics twice and could disagree without any test catching it,
# because each had its own regression suite.
#
# This file holds definitions only; both trainers `include` it.

import GeometricOptimizers

"""
    scalar_moment_adam_method(T; beta1, beta2, epsilon, ambient_norm)

The `scalar-moment-adam` method for parameters of element type `T`: `ScalarMomentAdam` on the
manifold leaves, ordinary `Adam` on the rest, with the same coefficients on both.

`ambient_norm` chooses which ‖·‖² the scalar second moment accumulates — `true` is the faithful
[li2020efficient] Algorithm 2 ambient norm, `false` the `GeometricOptimizers` quotient-space norm and
the default. It is not stored here: the trainers write it into every run record's `second_moment`,
which is where a number is traced back to the mode that produced it.

The learning rate is *not* part of the method. In `GeometricOptimizers` a method supplies a direction
and the rate is the line search's, and the two leaf kinds scale the same number differently — the
scalar-moment direction has magnitude ≈ 1 in total where `Adam`'s has ≈ 1 per component — which is
why this baseline is tuned at its own rate and why that rate is the caller's to pass.
"""
function scalar_moment_adam_method(::Type{T}; beta1 = 9.0e-1, beta2 = 9.9e-1,
        epsilon = 1.0e-8, ambient_norm::Bool = false) where {T <: AbstractFloat}
    GeometricOptimizers.CompositeMethod(;
        manifold = GeometricOptimizers.ScalarMomentAdam(T;
            β₁ = T(beta1), β₂ = T(beta2), δ = T(epsilon), ambient_norm = ambient_norm),
        array = GeometricOptimizers.Adam(T; β₁ = T(beta1), β₂ = T(beta2), δ = T(epsilon)))
end

"""The `second_moment` a run record quotes for the baseline, which is the mode it actually ran."""
function scalar_moment_description(ambient_norm::Bool)
    ambient_norm ? "scalar (ambient norm)" : "scalar (quotient norm)"
end
