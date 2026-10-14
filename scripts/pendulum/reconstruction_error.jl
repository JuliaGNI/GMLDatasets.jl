# reconstruction_error.jl — the relative reconstruction error of a level set of the pendulum
#
# The one reconstruction error the pendulum scripts report (`latent_plot.jl`, `branch_report.jl`,
# `psd_baseline.jl`). For an energy H it is
#
#   e_rec(H) = sqrt(<‖Ψ^dec(Ψ^enc(x)) − x‖²>) / sqrt(<‖x‖²>),
#
# with <·> the mean over the complete level set H, i.e. over one period of the exact pendulum orbit at
# that energy, sampled at `n` points uniformly in time (the way the training trajectories sample it)
# and lifted to ℝ⁴ by `angular_to_euclidean`. Above the separatrix each energy carries two rotating
# orbits; `sgn` picks the one whose p_θ has that sign. The numerator and the denominator are
# root-mean-squares over the same points, so `rel` is a ratio of norms of the whole level set, not a
# mean of pointwise ratios; `spread` is the standard deviation of the pointwise ratio
# ‖Ψ^dec(Ψ^enc(x)) − x‖/‖x‖ over the points.
#
# Time-uniform rather than angle-uniform: the drawing and the area computations of the scripts
# parametrize a level set by its own angle, which suits curves and line integrals; an average over
# states, like the training loss, should weight each part of the orbit by the time the motion spends
# there.
#
# The orbit is integrated with Gauss(2) of GeometricIntegrators at step T/n, T the period from the
# complete elliptic integral of the first kind; at n = N_ORBIT that is exact far below the digits the
# scripts print, and doubling n leaves those digits unchanged.

using GeometricIntegrators
import GMLDatasets: angular_to_euclidean

"Points per level set for `reconstruction_error`."
const N_ORBIT = 2000

# The complete elliptic integral K(m) = ∫₀^{π/2} (1 − m sin²s)^{-1/2} ds by the arithmetic-geometric
# mean, K(m) = π / (2 agm(1, √(1 − m))); no package in the environment provides it.
function elliptic_k(m)
    a, b = 1.0, sqrt(1 - m)
    while abs(a - b) > 4eps(a)
        a, b = (a + b) / 2, sqrt(a * b)
    end
    π / (a + b)
end

"""
    pendulum_period(H)

The period of the pendulum H = p_θ²/2 + cos θ: 4K((1+H)/2) on a librating level, and the time of one
turn, 4K(2/(1+H))/√(2(1+H)), on a rotating one.
"""
pendulum_period(H) = H < 1 ? 4elliptic_k((1 + H) / 2) : 4elliptic_k(2 / (1 + H)) / sqrt(2(1 + H))

"""
    orbit(H; n = N_ORBIT, sgn = -1) -> (θ, p_θ)

The exact orbit at energy `H`, `n` points uniformly in time over one period, starting at the top of
the libration (θ = π) or at θ = 0 on the rotating orbit with the sign `sgn` of p_θ.
"""
function orbit(H; n = N_ORBIT, sgn = -1)
    θ0, p0 = H < 1 ? (Float64(π), 2sqrt((1 + H) / 2)) : (0.0, sgn * sqrt(2(H - 1)))
    T = pendulum_period(H)
    v(v, t, q, p, params) = (v[1] = p[1]; v)
    f(f, t, q, p, params) = (f[1] = sin(q[1]); f)
    h(t, q, p, params) = p[1]^2 / 2 + cos(q[1])
    sol = integrate(HODEProblem(v, f, h, (0.0, T), T / n, [θ0], [p0]), Gauss(2))
    ([sol.q[i][1] for i in 0:(n - 1)], [sol.p[i][1] for i in 0:(n - 1)])
end

"""
    reconstruction_error(enc, dec, H; n = N_ORBIT, sgn = -1, T = Float32) -> (; rel, spread)

The relative reconstruction error of the level set `H` (see the header of this file). The states are
passed to the network as `T`, the reconstruction compared in Float64.
"""
function reconstruction_error(enc, dec, H::Real; n = N_ORBIT, sgn = -1, T = Float32)
    q, p = angular_to_euclidean(orbit(H; n, sgn)...)
    X = eachcol(vcat(q, p))
    e2 = [sum(abs2, Float64.(dec(enc(T.(x)))) .- x) for x in X]
    x2 = [sum(abs2, x) for x in X]
    r = sqrt.(e2 ./ x2)
    m = sum(r) / length(r)
    (rel = sqrt(sum(e2) / sum(x2)), spread = sqrt(sum(abs2, r .- m) / (length(r) - 1)))
end
